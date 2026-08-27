"""Generic ego-checkpoint loader for live submissions.

Pipeline:
  1. User uploads a tar.gz/zip with a saved_train_run/ directory inside
     (orbax checkpoint format produced by common.save_load_utils.save_train_run).
  2. We extract to a temp dir.
  3. The user specifies actor_type (mlp/s5/rnn) and arch params; we build
     a policy network via agents.initialize_agents on the right env.
  4. We load checkpoint params via common.save_load_utils.load_checkpoints.
  5. Return a (obs, state, rng) -> int callable that the eval pipeline
     can use as ego_fn.

Cleanup of temp dirs is the caller's responsibility (we yield the path
and let the route delete after eval).
"""
from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class UploadedCheckpoint:
    """Parsed metadata + extracted location of a user-uploaded ego ckpt."""
    extracted_dir: Path
    saved_train_run_path: Path
    actor_type: str
    arch_params: dict[str, Any]
    ckpt_key: str
    idx: int


class CheckpointParseError(ValueError):
    pass


def extract_archive(archive_bytes: bytes, dest_root: Path | None = None) -> Path:
    """Extract a tar.gz or zip into a fresh temp dir, return path."""
    dest = Path(tempfile.mkdtemp(prefix="bm_ego_", dir=str(dest_root) if dest_root else None))
    log.info("extracting upload to %s", dest)

    sniff = archive_bytes[:8]
    is_zip = sniff[:4] == b"PK\x03\x04" or sniff[:4] == b"PK\x05\x06"
    is_tar_gz = sniff[:2] == b"\x1f\x8b"
    is_tar = sniff[257:262] == b"ustar" if len(archive_bytes) > 262 else False

    archive_path = dest / "_upload.bin"
    archive_path.write_bytes(archive_bytes)

    if is_zip:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif is_tar_gz or is_tar:
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    else:
        raise CheckpointParseError(
            "upload must be .zip or .tar.gz with a saved_train_run/ directory inside"
        )

    archive_path.unlink(missing_ok=True)
    return dest


def find_saved_train_run(extracted_dir: Path) -> Path:
    """Locate the saved_train_run/ directory in an extracted archive."""
    candidate = extracted_dir / "saved_train_run"
    if candidate.is_dir():
        return candidate
    for sub in extracted_dir.rglob("saved_train_run"):
        if sub.is_dir():
            return sub
    raise CheckpointParseError(
        "no saved_train_run/ directory found in upload. "
        "expected the orbax checkpoint format from common.save_load_utils.save_train_run"
    )


def parse_upload(
    archive_bytes: bytes,
    actor_type: str,
    arch_params: dict[str, Any] | None = None,
    ckpt_key: str = "final_params",
    idx: int = 0,
) -> UploadedCheckpoint:
    """Extract the upload + validate it has the expected structure."""
    extracted = extract_archive(archive_bytes)
    saved_path = find_saved_train_run(extracted)
    return UploadedCheckpoint(
        extracted_dir=extracted,
        saved_train_run_path=saved_path,
        actor_type=actor_type,
        arch_params=dict(arch_params or {}),
        ckpt_key=ckpt_key,
        idx=idx,
    )


def cleanup_upload(uploaded: UploadedCheckpoint) -> None:
    """Best-effort cleanup of the temp extraction dir."""
    try:
        shutil.rmtree(uploaded.extracted_dir)
    except OSError as exc:
        log.warning("failed to clean up %s: %s", uploaded.extracted_dir, exc)


def build_ego_fn(env_name: str, env, uploaded: UploadedCheckpoint) -> Callable:
    """Construct (obs, state, rng) -> int from an extracted ckpt.

    Delegates to env adapter's load_ego_checkpoint if defined, otherwise
    uses the generic policy-class dispatch in agents.initialize_agents.
    """
    from .envs import get as get_env

    adapter = get_env(env_name)
    custom_loader = getattr(adapter, "load_ego_checkpoint", None)
    if callable(custom_loader):
        return custom_loader(
            saved_train_run_path=str(uploaded.saved_train_run_path),
            actor_type=uploaded.actor_type,
            arch_params=uploaded.arch_params,
            ckpt_key=uploaded.ckpt_key,
            idx=uploaded.idx,
            env=env,
        )
    return _generic_load_ego(env, uploaded)


def _generic_load_ego(env, uploaded: UploadedCheckpoint) -> Callable:
    """Default ego loader: constructs policy via agents.initialize_agents
    based on actor_type, then loads weights via common.save_load_utils.

    Used when the env adapter doesn't override load_ego_checkpoint.
    """
    import jax

    from agents.initialize_agents import (
        initialize_s5_agent, initialize_mlp_agent, initialize_rnn_agent,
    )
    from common.save_load_utils import load_checkpoints

    actor_type = uploaded.actor_type.lower()
    cfg = dict(uploaded.arch_params)

    # Some envs (Hanabi) expose observation_space.shape as a scalar int instead
    # of a tuple. initialize_*_agent does `.shape[0]` which errors on scalar shapes,
    # AND dict.get evaluates the default eagerly so injecting POLICY_INPUT_DIM
    # doesn't bypass the broken expression. Patch the Box's shape to a tuple
    # in-place so the lookup succeeds for both branches.
    import numpy as np
    sp = env.observation_space(env.agents[0])
    shape = getattr(sp, "shape", None)
    if shape is not None and not isinstance(shape, tuple):
        try:
            sp.shape = (int(shape),) if np.isscalar(shape) else tuple(int(x) for x in np.atleast_1d(shape))
        except Exception:
            cfg.setdefault("POLICY_INPUT_DIM", int(shape) if np.isscalar(shape) else int(np.prod(shape)))

    rng = jax.random.PRNGKey(0)
    if actor_type == "s5":
        policy, _ = initialize_s5_agent(cfg, env, rng)
    elif actor_type == "mlp":
        policy, _ = initialize_mlp_agent(cfg, env, rng)
    elif actor_type == "rnn":
        policy, _ = initialize_rnn_agent(cfg, env, rng)
    else:
        raise CheckpointParseError(
            f"actor_type '{actor_type}' not supported by generic loader. "
            "supported: mlp, s5, rnn. for custom architectures, the env adapter "
            "must implement load_ego_checkpoint."
        )

    params = load_checkpoints(
        str(uploaded.saved_train_run_path),
        ckpt_key=uploaded.ckpt_key,
    )
    if isinstance(params, list):
        params = params[uploaded.idx]
    elif hasattr(params, "shape") or isinstance(params, dict):
        try:
            params = jax.tree.map(lambda x: x[uploaded.idx], params)
        except Exception:
            pass

    hstate_holder = [None]

    def ego_fn(obs, state, rng):
        import jax.numpy as jnp
        if hstate_holder[0] is None:
            try:
                hstate_holder[0] = policy.init_hstate(1, aux_info={"agent_id": 0})
            except Exception:
                hstate_holder[0] = None

        if isinstance(obs, dict):
            agent_obs = obs["agent_0"] if "agent_0" in obs else next(iter(obs.values()))
        else:
            agent_obs = obs

        try:
            avail_dim = int(env.action_space(env.agents[0]).n)
        except Exception:
            avail_dim = 21

        action, new_hstate = policy.get_action_value_policy(
            params=params,
            obs=jnp.asarray(agent_obs).reshape(1, 1, -1),
            done=jnp.zeros((1, 1), dtype=bool),
            avail_actions=jnp.ones((1, 1, avail_dim), dtype=jnp.float32),
            hstate=hstate_holder[0],
            rng=rng,
        )[:2] if hasattr(policy, "get_action_value_policy") else (None, None)
        if action is None:
            try:
                action, new_hstate = policy.get_action(
                    params=params, obs=agent_obs,
                    done=jnp.array(False),
                    avail_actions=jnp.ones((avail_dim,), dtype=jnp.float32),
                    hstate=hstate_holder[0], rng=rng,
                    env_state=state,
                )
            except Exception:
                raise

        hstate_holder[0] = new_hstate
        return int(jnp.asarray(action).reshape(-1)[0])

    return ego_fn
