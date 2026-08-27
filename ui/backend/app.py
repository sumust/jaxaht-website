"""ui Flask app.

Routes are thin: validate payload via Pydantic, dispatch to the right
EnvRenderer via the registry, call into the SessionManager or storage
layer, return a Pydantic-shaped response.

Run:
    python -m ui.backend.app         # dev server on :5174
or:
    flask --app ui.backend.app run
"""
from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from pydantic import ValidationError

from .envs import get as get_env, registry as env_registry
from .eval import evaluate_full_suite
from .helpers import current_player as _current_player, next_rng as _next_rng, scrub as _scrub
from .heldout_loader import list_versions, load_heldout_partners
from .jobs import JobManager, STATUS_DONE
from .schemas import (
    AggregateScore,
    BuiltinEgoInfo,
    BuiltinEgosResponse,
    EnvInfo,
    EnvsResponse,
    ErrorResponse,
    HeldoutPartnerInfo,
    HeldoutVersionInfo,
    JobProgress,
    JobStatusResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    NewGameRequest,
    NewGameResponse,
    PartnerInfo,
    PartnersResponse,
    PerPartnerScore,
    SaveTrajectoryRequest,
    SaveTrajectoryResponse,
    StepRequest,
    StepResponse,
    SubmitRequest,
    SubmitResponse,
    ClientResultSubmission,
)
from .sessions import SessionManager
from .storage import build_backends
from .study_session import ProlificMeta, StudySessionManager

log = logging.getLogger("ui")


def create_app(backends=None) -> Flask:
    """Factory so tests can inject their own backends."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    backends = backends or build_backends()
    app.extensions["backends"] = backends
    sm = SessionManager(backends.session)
    # Background daemon that purges sessions idle > 1 hour every 5 min.
    # Daemon so it doesn't block shutdown. Disabled in test fixtures.
    if os.environ.get("DISABLE_PURGE_THREAD") != "1":
        sm.start_purge_thread(ttl_seconds=3600, interval_seconds=300)
    app.extensions["sessions"] = sm
    app.extensions["studies"] = StudySessionManager(sm)
    app.extensions["jobs"] = JobManager(backends.job, max_workers=2)

    _register_routes(app)
    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ValidationError)
    def _bad_request(exc: ValidationError):
        return jsonify(ErrorResponse(
            error="validation_error",
            detail=str(exc),
        ).model_dump()), 400

    @app.errorhandler(KeyError)
    def _not_found(exc: KeyError):
        return jsonify(ErrorResponse(
            error="not_found",
            detail=str(exc).strip("'"),
        ).model_dump()), 404

    @app.errorhandler(ValueError)
    def _value_error(exc: ValueError):
        return jsonify(ErrorResponse(
            error="bad_request",
            detail=str(exc),
        ).model_dump()), 400


def _register_routes(app: Flask) -> None:  # noqa: C901 (route fan-out)

    @app.get("/api/healthz")
    def healthz():
        return jsonify({"ok": True, "envs": sorted(env_registry().keys())})

    @app.get("/api/envs")
    def list_envs():
        # Hidden envs (mock/dev harnesses) are omitted from the public
        # Home list but still reachable via direct /api/<env>/... routes
        # so tests and manual curls keep working.
        include_hidden = request.args.get("include_hidden") == "true"
        envs = []
        for name, adapter in env_registry().items():
            if getattr(adapter, "HIDDEN", False) and not include_hidden:
                continue
            partners = adapter.available_partners() if adapter.READY else []
            envs.append(EnvInfo(
                env_name=name,
                display_name=getattr(adapter, "DISPLAY_NAME", name.title()),
                modes=[m.key for m in adapter.modes() if m.enabled],
                default_partner_key=adapter.default_partner_key() if partners else "",
                overview=getattr(adapter, "OVERVIEW", ""),
                stats=dict(getattr(adapter, "STATS", {})),
                accent=getattr(adapter, "ACCENT", "blue"),
                ready=bool(getattr(adapter, "READY", True)),
                num_partners=len(partners),
            ))
        return jsonify(EnvsResponse(envs=envs).model_dump())

    @app.get("/api/<env>/partners")
    def list_partners(env: str):
        adapter = get_env(env)
        partners = [
            PartnerInfo(
                key=p.key,
                display_name=p.display_name,
                difficulty=p.difficulty,
                description=p.description,
                tags=p.tags,
            )
            for p in adapter.available_partners()
        ]
        return jsonify(PartnersResponse(partners=partners).model_dump())

    @app.get("/api/<env>/controls")
    def get_controls(env: str):
        # Per-env keyboard map (action_id, name) so the frontend can render
        # a controls legend without hardcoding env-specific knowledge.
        # Mirrors Johnny's human_data_collecting /api/controls endpoint.
        adapter = get_env(env)
        kb = adapter.keyboard_controls() if hasattr(adapter, "keyboard_controls") else {}
        action_to_name = {
            "noop": "No-op (wait)",
            "up": "Move up", "down": "Move down",
            "left": "Move left", "right": "Move right",
            "load": "Load / collect",
            "stay": "Stay", "interact": "Interact (pickup, drop, plate, serve)",
            "discard": "Discard a card", "play": "Play a card",
            "hint_color": "Hint a color", "hint_rank": "Hint a rank",
        }
        keyboard = {}
        for key, action_id in (kb or {}).items():
            label_key = key if key != " " else "space"
            keyboard[label_key] = {"action": int(action_id) if isinstance(action_id, (int, str)) and str(action_id).lstrip("-").isdigit() else action_id,
                                   "name": action_to_name.get(str(action_id), str(action_id))}
        return jsonify({
            "env": env,
            "keyboard": keyboard,
            "actions": action_to_name,
        })

    @app.post("/api/<env>/play/new")
    def new_game(env: str):
        payload = NewGameRequest.model_validate(request.get_json(force=True))
        sm: SessionManager = app.extensions["sessions"]
        live, obs = sm.start(env, payload.partner_key, payload.env_kwargs, payload.seed)
        resp = NewGameResponse(
            session_id=live.session_id,
            state=live.renderer.serialize_state(live.state, obs),
            score=live.renderer.score_summary(live.state, {}),
        )
        return jsonify(resp.model_dump())

    @app.post("/api/<env>/play/step")
    def step(env: str):
        payload = StepRequest.model_validate(request.get_json(force=True))
        sm: SessionManager = app.extensions["sessions"]
        live = sm.get(payload.session_id)
        if live.env_name != env:
            raise ValueError(
                f"session {payload.session_id} is for env '{live.env_name}', not '{env}'"
            )
        renderer = live.renderer
        human_idx = renderer.HUMAN_AGENT_IDX
        partner_idx = 1 - human_idx
        action = renderer.action_from_ui(payload.action, live.state)
        events: list[dict] = []
        partner_acted = False

        if renderer.IS_TURN_BASED:
            # turn-based (Hanabi): only current_player acts; partner acts after if it's their turn next.
            current_player = _current_player(live.state, human_idx)
            if current_player != human_idx:
                raise ValueError("not the human's turn; call /step_partner first")

            rng = _next_rng()
            state_before_human = live.state
            obs, state, reward, done, info = renderer.step(
                live.env, live.state, {human_idx: action}, rng,
            )
            live.state = state
            human_event = renderer.describe_action(
                action, state_before_human, live.state, float(reward), human_idx,
            )
            events.append(human_event)
            live.history.append(human_event)

            if not done:
                current_player = _current_player(live.state, human_idx)
                if current_player == partner_idx:
                    # ask partner from a fresh obs (post-human-action) before stepping again
                    partner_action = int(live.partner.get_action(obs, live.state, _next_rng()))
                    state_before_partner = live.state
                    obs, state, p_reward, done, info = renderer.step(
                        live.env, live.state, {partner_idx: partner_action}, _next_rng(),
                    )
                    live.state = state
                    reward += float(p_reward)
                    partner_event = renderer.describe_action(
                        partner_action, state_before_partner, live.state,
                        float(p_reward), partner_idx,
                    )
                    events.append(partner_event)
                    live.history.append(partner_event)
                    partner_acted = True
        else:
            # simultaneous (LBF, Overcooked): both act every step.
            partner_action = int(live.partner.get_action(live.last_obs, live.state, _next_rng()))
            state_before = live.state
            obs, state, reward, done, info = renderer.step(
                live.env, live.state,
                {human_idx: action, partner_idx: partner_action},
                _next_rng(),
            )
            live.state = state
            human_event = renderer.describe_action(
                action, state_before, live.state, float(reward), human_idx,
            )
            partner_event = renderer.describe_action(
                partner_action, state_before, live.state, 0.0, partner_idx,
            )
            events.extend([human_event, partner_event])
            live.history.extend([human_event, partner_event])
            partner_acted = True

        live.last_obs = obs
        return jsonify(StepResponse(
            state=renderer.serialize_state(live.state, obs),
            score=renderer.score_summary(live.state, info),
            reward=float(reward),
            done=bool(done),
            info=_scrub(info),
            partner_acted=partner_acted,
            events=events,
        ).model_dump())

    @app.post("/api/<env>/demo/upload")
    def demo_upload(env: str):
        # Upload a checkpoint for policy demo only. Stores the ckpt, registers
        # a leaderboard entry tagged kind="demo" so it appears in uploaded_partners
        # but doesn't pollute the eval leaderboard. No evaluation runs.
        if not request.content_type or not request.content_type.startswith("multipart/form-data"):
            raise ValueError("multipart/form-data required: payload (json) + checkpoint (file)")
        import json as _json
        payload_raw = request.form.get("payload")
        if not payload_raw:
            raise ValueError("payload (json) field required")
        payload = _json.loads(payload_raw)
        upload_file = request.files.get("checkpoint")
        if upload_file is None:
            raise ValueError("checkpoint file required")
        uploaded_bytes = upload_file.read()

        actor_type = payload.get("actor_type")
        if not actor_type:
            raise ValueError("actor_type required (mlp|s5|rnn)")
        agent_name = payload.get("agent_name") or "Uploaded policy"
        arch_params = payload.get("arch_params") or {}

        backends = app.extensions["backends"]
        ckpt_id = backends.checkpoint.save_upload(uploaded_bytes, {
            "agent_name": agent_name, "env": env,
            "actor_type": actor_type, "arch_params": arch_params,
            "kind": "demo",
        })
        entry = {
            "env": env, "version": "v1",
            "agent_name": agent_name, "ego_kind": "upload",
            "actor_type": actor_type, "arch_params": arch_params,
            "checkpoint_id": ckpt_id,
            "checkpoint_sha256": backends.checkpoint.get_sha256(ckpt_id),
            "kind": "demo",
            "num_episodes": 0, "eval_seed": 0,
            "aggregate_score": 0.0, "aggregate": None,
            "per_partner": [],
        }
        entry_id = backends.leaderboard.add_entry(env, "v1", entry)
        return jsonify({"checkpoint_id": ckpt_id, "entry_id": entry_id, "ok": True})

    @app.get("/api/<env>/uploaded_partners")
    def list_uploaded_partners(env: str):
        # return ckpts uploaded for this env that can be selected as a demo agent.
        # backed by leaderboard entries with ego_kind="upload" + checkpoint_id set.
        backends = app.extensions["backends"]
        entries = backends.leaderboard.list_entries(env, "v1")
        out = []
        seen_ids = set()
        for e in entries:
            ckpt_id = e.get("checkpoint_id")
            if not ckpt_id or ckpt_id in seen_ids:
                continue
            if e.get("ego_kind") != "upload":
                continue
            seen_ids.add(ckpt_id)
            out.append({
                "checkpoint_id": ckpt_id,
                "agent_name": e.get("agent_name", "Uploaded ego"),
                "actor_type": e.get("actor_type"),
                "aggregate_score": e.get("aggregate_score"),
                "created_at": e.get("created_at"),
                "checkpoint_sha256": e.get("checkpoint_sha256"),
            })
        return jsonify({"uploaded": out})

    @app.post("/api/<env>/play/demo")
    def demo(env: str):
        payload_json = request.get_json(force=True) or {}
        # agent_*_id is prefixed: "partner:<key>" or "ego:<key>".
        # Bare keys + the older partner_*_key fields fall back to partner: for back-compat.
        agent_a_id = (
            payload_json.get("agent_a_id")
            or payload_json.get("partner_a_key")
            or payload_json.get("partner_key")
        )
        agent_b_id = (
            payload_json.get("agent_b_id")
            or payload_json.get("partner_b_key")
            or agent_a_id
        )
        if not agent_a_id:
            raise ValueError("agent_a_id (or partner_a_key) required")
        seed = int(payload_json.get("seed", 42))
        max_steps = int(payload_json.get("max_steps", 100))

        renderer = get_env(env)
        partners = {p.key: p for p in renderer.available_partners()}
        egos = {s.key: s for s in renderer.builtin_egos()}

        def _resolve(agent_id: str):
            if ":" in agent_id:
                kind, key = agent_id.split(":", 1)
            else:
                kind, key = "partner", agent_id
            if kind == "partner":
                if key not in partners:
                    raise ValueError(f"unknown partner '{key}' for env '{env}'")
                return partners[key].load_fn()
            if kind == "ego":
                if key not in egos:
                    raise ValueError(f"unknown builtin ego '{key}' for env '{env}'")
                return egos[key].load_fn()
            if kind == "upload":
                # load an uploaded ckpt as a demo agent. Look up the leaderboard
                # entry to find its actor_type + arch_params, then rehydrate.
                backends = app.extensions["backends"]
                entries = backends.leaderboard.list_entries(env, "v1")
                entry = next((e for e in entries if e.get("checkpoint_id") == key), None)
                if entry is None:
                    raise ValueError(f"upload ckpt '{key}' not found in leaderboard for {env}")
                ckpt_path = backends.checkpoint.get_path(key)
                if ckpt_path is None:
                    # try to lazy-fetch from the HF dataset (ckpts aren't bootstrapped eagerly)
                    hf_repo = os.environ.get("BENCHMARK_HF_REPO")
                    data_root = os.environ.get("DATA_ROOT", "./ui/data")
                    if hf_repo:
                        from .storage.hf_mirror import ensure_checkpoint_local
                        ensure_checkpoint_local(hf_repo, key, data_root)
                        ckpt_path = backends.checkpoint.get_path(key)
                    if ckpt_path is None:
                        raise ValueError(f"upload ckpt '{key}' has no extracted dir on disk")
                from .checkpoint_loader import find_saved_train_run, UploadedCheckpoint, build_ego_fn
                from pathlib import Path
                saved = find_saved_train_run(Path(ckpt_path))
                uploaded = UploadedCheckpoint(
                    extracted_dir=Path(ckpt_path),
                    saved_train_run_path=saved,
                    actor_type=entry.get("actor_type") or "mlp",
                    arch_params=entry.get("arch_params") or {},
                    ckpt_key=entry.get("ckpt_key", "final_params"),
                    idx=entry.get("idx", 0),
                )
                ego_fn = build_ego_fn(env_name=env, env=env_obj, uploaded=uploaded)
                # wrap into the get_action(obs, state, rng) -> int contract demo expects
                class _UploadAdapter:
                    def __init__(self, fn): self.fn = fn
                    def get_action(self, obs, state, rng):
                        return int(self.fn(obs, state, rng))
                return _UploadAdapter(ego_fn)
            raise ValueError(f"unsupported agent kind '{kind}' (got id '{agent_id}')")

        env_obj = renderer.make_env({})
        rng = _next_rng()
        obs, state = renderer.reset(env_obj, rng)
        agents = [_resolve(agent_a_id), _resolve(agent_b_id)]

        # Some envs wrap state in WrappedEnvState; serialize_state expects
        # the inner env_state. Unwrap before passing.
        def _inner(s):
            return getattr(s, "env_state", s)

        frames = [{
            "state": renderer.serialize_state(_inner(state), obs),
            "score": renderer.score_summary(_inner(state), {}),
            "step": 0,
            "actor": None,
            "action": None,
            "reward": 0.0,
            "event": None,
        }]
        total_reward = 0.0
        last_info: dict = {}
        done = False
        for step_i in range(max_steps):
            actions: dict[int, int] = {}
            for idx in (0, 1):
                try:
                    actions[idx] = int(agents[idx].get_action(obs, state, _next_rng()))
                except Exception as exc:
                    log.warning("demo: agent %d get_action failed at step %d: %s", idx, step_i, exc)
                    actions[idx] = 0
            cur_player = _current_player(_inner(state), 0)
            state_before = state
            obs, state, reward, done, info = renderer.step(env_obj, state, actions, _next_rng())
            total_reward += float(reward)
            last_info = info
            new_state = renderer.serialize_state(_inner(state), obs)
            new_score = renderer.score_summary(_inner(state), info)
            if renderer.IS_TURN_BASED:
                # one frame per turn — only cur_player acted this step.
                event = renderer.describe_action(
                    actions[cur_player], _inner(state_before), _inner(state), float(reward), cur_player,
                )
                frames.append({
                    "state": new_state, "score": new_score,
                    "step": step_i + 1, "actor": cur_player, "action": actions[cur_player],
                    "reward": float(reward), "event": event, "done": bool(done),
                })
            else:
                # simultaneous — both agents acted; emit one frame per agent with same
                # post-step state so the UI can attribute moves and play them back.
                for p in (0, 1):
                    event_p = renderer.describe_action(
                        actions[p], _inner(state_before), _inner(state), float(reward) if p == 0 else 0.0, p,
                    )
                    frames.append({
                        "state": new_state, "score": new_score,
                        "step": step_i + 1, "actor": p, "action": actions[p],
                        "reward": float(reward) if p == 0 else 0.0,
                        "event": event_p, "done": bool(done) if p == 1 else False,
                    })
            if done:
                break

        return jsonify({
            "frames": frames,
            "total_reward": total_reward,
            "final_score": renderer.score_summary(_inner(state), last_info),
            "done": bool(done),
            "num_frames": len(frames),
            "agent_a_id": agent_a_id,
            "agent_b_id": agent_b_id,
        })

    @app.post("/api/<env>/play/save")
    def save_trajectory(env: str):
        payload = SaveTrajectoryRequest.model_validate(request.get_json(force=True))
        sm: SessionManager = app.extensions["sessions"]
        live = sm.get(payload.session_id)
        backends = app.extensions["backends"]
        traj = {
            "env": env,
            "agent_name": payload.agent_name,
            "partner_key": live.partner_key,
            "score": live.renderer.score_summary(live.state, {}),
            "history": live.history,
        }
        tid = backends.trajectory.save(env, traj)
        return jsonify(SaveTrajectoryResponse(trajectory_id=tid).model_dump())

    # --------- prolific study flow (multi-game sessions) ---------

    @app.post("/api/<env>/study/start")
    def study_start(env: str):
        get_env(env)
        payload = request.get_json(force=True) or {}
        # Anti-path-traversal: data_source is used in folder names, so restrict
        # to alphanumeric + hyphens. Matches Johnny's sanitization in human_data_collecting.
        raw_source = (payload.get("data_source") or "").strip()
        data_source = raw_source if all(c.isalnum() or c == "-" for c in raw_source) else ""
        prolific = ProlificMeta(
            prolific_pid=payload.get("prolific_pid"),
            study_id=payload.get("study_id"),
            prolific_session_id=payload.get("prolific_session_id"),
            data_source=data_source,
        )
        studies: StudySessionManager = app.extensions["studies"]
        study = studies.start(env, prolific)
        # Return first game's state so the client can render immediately.
        sm: SessionManager = app.extensions["sessions"]
        live = sm.get(study.game_session_ids[0])
        return jsonify({
            "study_id": study.study_id,
            "current_game_index": 0,
            "total_games": study.config.total_games(),
            "num_warmup": study.config.num_warmup,
            "is_warmup": study.is_warmup(),
            "session_id": live.session_id,
            "state": live.renderer.serialize_state(live.state, None),
            "score": live.renderer.score_summary(live.state, {}),
            "partner_key": live.partner_key,
        })

    @app.post("/api/<env>/study/step")
    def study_step(env: str):
        get_env(env)
        payload = request.get_json(force=True) or {}
        study_id = payload.get("study_id")
        ui_action = payload.get("action") or {}
        if not study_id:
            raise ValueError("study_id required")
        studies: StudySessionManager = app.extensions["studies"]
        result = studies.step(study_id, ui_action)
        # Attach current LiveSession state for rendering.
        sm: SessionManager = app.extensions["sessions"]
        study = studies.get(study_id)
        sid = study.current_game_id()
        if sid:
            live = sm.get(sid)
            result["session_id"] = live.session_id
            result["state"] = live.renderer.serialize_state(live.state, None)
            result["score"] = live.renderer.score_summary(live.state, {})
            result["partner_key"] = live.partner_key
        return jsonify(result)

    @app.post("/api/<env>/study/save")
    def study_save(env: str):
        get_env(env)
        payload = request.get_json(force=True) or {}
        study_id = payload.get("study_id")
        agent_name = payload.get("agent_name", "Anonymous")
        if not study_id:
            raise ValueError("study_id required")
        studies: StudySessionManager = app.extensions["studies"]
        backends = app.extensions["backends"]
        trajs = studies.save_all(study_id, agent_name)
        ids = [backends.trajectory.save(env, t) for t in trajs]
        return jsonify({"trajectory_ids": ids, "count": len(ids)})

    @app.post("/api/<env>/submit/result")
    def submit_client_result(env: str):
        # client-eval path: caller already ran heldout eval locally and
        # is posting the result row. no checkpoint upload, no job queue.
        get_env(env)
        payload = ClientResultSubmission.model_validate(request.get_json(force=True))
        backends = app.extensions["backends"]
        entry = {
            "env": env,
            "version": payload.version,
            "agent_name": payload.agent_name,
            "ego_kind": "client_eval",
            "checkpoint_sha256": payload.checkpoint_sha256,
            "num_episodes": payload.num_episodes,
            "eval_seed": payload.eval_seed,
            "notes": payload.notes,
            "aggregate_score": payload.aggregate.mean,
            "aggregate": payload.aggregate.model_dump(),
            "per_partner": [p.model_dump() for p in payload.per_partner],
            "wall_clock_seconds": payload.wall_clock_seconds,
        }
        entry_id = backends.leaderboard.add_entry(env, payload.version, entry)
        return jsonify({"entry_id": entry_id})

    @app.get("/prolific/start")
    def prolific_landing():
        """Env-agnostic Prolific landing. Reads PROLIFIC_PID/STUDY_ID/
        SESSION_ID from the URL and 302-redirects to the React study page
        for the given env. Cookie pattern keeps the IDs across requests."""
        env = request.args.get("env")
        if not env:
            raise ValueError("env required")
        get_env(env)
        prolific_pid = request.args.get("PROLIFIC_PID", "")
        study_id_param = request.args.get("STUDY_ID", "")
        session_id_param = request.args.get("SESSION_ID", "")
        # Forward params to the React app via query string.
        from urllib.parse import urlencode
        qs = urlencode({
            "PROLIFIC_PID": prolific_pid,
            "STUDY_ID": study_id_param,
            "SESSION_ID": session_id_param,
        })
        # The dev/vite proxy + prod static serve both root the SPA at /; the
        # SPA will route this URL on the client.
        from flask import redirect
        return redirect(f"/{env}/study?{qs}", code=302)

    # --------- heldout + builtin egos ---------

    @app.get("/api/<env>/heldout")
    def list_heldout(env: str):
        get_env(env)   # raises KeyError -> 404
        version = request.args.get("version", "v1")
        partners = load_heldout_partners(env, version)
        payload = HeldoutVersionInfo(
            env=env,
            version=version,
            partners=[
                HeldoutPartnerInfo(
                    key=p.key,
                    display_name=p.display_name,
                    difficulty=p.difficulty,
                    description=p.description,
                    tags=p.tags,
                    normalize_bounds=p.normalize_bounds,
                )
                for p in partners
            ],
        )
        return jsonify(payload.model_dump())

    @app.get("/api/<env>/heldout/versions")
    def heldout_versions(env: str):
        get_env(env)
        return jsonify({"versions": list_versions(env)})

    @app.get("/api/<env>/egos")
    def list_egos(env: str):
        adapter = get_env(env)
        egos = [
            BuiltinEgoInfo(
                key=spec.key,
                display_name=spec.display_name,
                description=spec.description,
                tags=spec.tags,
            )
            for spec in adapter.builtin_egos()
        ]
        return jsonify(BuiltinEgosResponse(egos=egos).model_dump())

    # --------- submit + leaderboard ---------

    @app.post("/api/<env>/submit")
    def submit(env: str):
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            import json as _json
            payload_raw = request.form.get("payload")
            if not payload_raw:
                raise ValueError("multipart submit must include a 'payload' JSON field")
            payload = SubmitRequest.model_validate(_json.loads(payload_raw))
            upload_file = request.files.get("checkpoint")
            uploaded_bytes = upload_file.read() if upload_file is not None else None
        else:
            payload = SubmitRequest.model_validate(request.get_json(force=True))
            uploaded_bytes = None

        adapter = get_env(env)
        ego_factory = None
        upload_holder: dict[str, Any] = {}
        backends = app.extensions["backends"]

        if payload.ego_kind == "builtin":
            if not payload.builtin_key:
                raise ValueError("ego_kind=builtin requires builtin_key")
            egos = {s.key: s for s in adapter.builtin_egos()}
            if payload.builtin_key not in egos:
                raise ValueError(f"unknown builtin ego '{payload.builtin_key}' for {env}")
            ego_spec = egos[payload.builtin_key]
            ego_factory = ego_spec.load_fn
            display_name = payload.agent_name or ego_spec.display_name
        elif payload.ego_kind == "upload":
            if uploaded_bytes is None:
                raise ValueError(
                    "ego_kind=upload requires a 'checkpoint' file in the multipart form. "
                    "POST as multipart/form-data with `payload` (json) and `checkpoint` (file)"
                )
            if not payload.actor_type:
                raise ValueError("ego_kind=upload requires actor_type (mlp|s5|rnn)")
            display_name = payload.agent_name or "Uploaded ego"
            from .checkpoint_loader import parse_upload, build_ego_fn, cleanup_upload
            uploaded = parse_upload(
                uploaded_bytes,
                actor_type=payload.actor_type,
                arch_params=payload.arch_params or {},
                ckpt_key=payload.ckpt_key,
                idx=payload.idx,
            )
            upload_holder["uploaded"] = uploaded
            # persist the raw upload so submissions survive container rebuilds
            ckpt_id = backends.checkpoint.save_upload(uploaded_bytes, {
                "agent_name": payload.agent_name,
                "env": env,
                "actor_type": payload.actor_type,
                "ckpt_key": payload.ckpt_key,
                "idx": payload.idx,
            })
            upload_holder["ckpt_id"] = ckpt_id

            def ego_factory():
                env_obj = adapter.make_env(adapter.DEFAULT_KWARGS or {})
                return build_ego_fn(env_name=env, env=env_obj, uploaded=uploaded)
        else:
            raise ValueError(f"unknown ego_kind '{payload.ego_kind}'")

        jobs: JobManager = app.extensions["jobs"]

        def _run(progress_cb):
            try:
                ego_fn = ego_factory()
                result = evaluate_full_suite(
                    env_name=env,
                    version=payload.version,
                    ego_fn=ego_fn,
                    num_episodes=payload.num_episodes,
                    seed=payload.eval_seed,
                    progress_cb=progress_cb,
                )
                entry = {
                    "env": env,
                    "version": payload.version,
                    "agent_name": display_name,
                    "ego_kind": payload.ego_kind,
                    "builtin_key": payload.builtin_key,
                    "actor_type": payload.actor_type,
                    "num_episodes": payload.num_episodes,
                    "eval_seed": payload.eval_seed,
                    "aggregate_score": result["aggregate"]["mean"],
                    "aggregate": result["aggregate"],
                    "per_partner": result["per_partner"],
                    "wall_clock_seconds": result["wall_clock_seconds"],
                    "notes": payload.notes,
                    "checkpoint_id": upload_holder.get("ckpt_id"),
                    "checkpoint_sha256": (
                        backends.checkpoint.get_sha256(upload_holder["ckpt_id"])
                        if "ckpt_id" in upload_holder else None
                    ),
                }
                entry_id = backends.leaderboard.add_entry(env, payload.version, entry)
                return {"entry_id": entry_id, **result}
            finally:
                if "uploaded" in upload_holder:
                    from .checkpoint_loader import cleanup_upload
                    cleanup_upload(upload_holder["uploaded"])

        job_id = jobs.submit(
            kind="evaluate",
            env=env,
            params=payload.model_dump(),
            fn=_run,
        )
        return jsonify(SubmitResponse(job_id=job_id).model_dump())

    @app.get("/api/<env>/submit/status/<job_id>")
    def submit_status(env: str, job_id: str):
        jobs: JobManager = app.extensions["jobs"]
        job = jobs.get(job_id)
        if job is None or job.get("env") != env:
            raise KeyError(f"job {job_id}")
        return jsonify(JobStatusResponse(
            id=job["id"],
            kind=job.get("kind", ""),
            env=job["env"],
            status=job["status"],
            progress=JobProgress(**job.get("progress", {"completed": 0, "total": 0, "current": None})),
            error=job.get("error"),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            started_at=job.get("started_at"),
            finished_at=job.get("finished_at"),
        ).model_dump())

    @app.get("/api/<env>/submit/result/<job_id>")
    def submit_result(env: str, job_id: str):
        jobs: JobManager = app.extensions["jobs"]
        job = jobs.get(job_id)
        if job is None or job.get("env") != env:
            raise KeyError(f"job {job_id}")
        if job["status"] != STATUS_DONE:
            return jsonify({"status": job["status"], "error": job.get("error")}), 202
        return jsonify(job["result"])

    @app.get("/api/<env>/leaderboard")
    def leaderboard(env: str):
        get_env(env)
        version = request.args.get("version", "v1")
        entries = app.extensions["backends"].leaderboard.list_entries(env, version)
        # demo-only uploads live in the same store but shouldn't appear on the eval leaderboard
        entries = [e for e in entries if e.get("kind") != "demo"]
        return jsonify(LeaderboardResponse(
            env=env,
            version=version,
            entries=[_entry_to_model(e) for e in entries],
        ).model_dump())

    @app.get("/api/<env>/leaderboard/<entry_id>")
    def leaderboard_entry(env: str, entry_id: str):
        version = request.args.get("version", "v1")
        entry = app.extensions["backends"].leaderboard.get_entry(env, version, entry_id)
        if entry is None:
            raise KeyError(f"entry {entry_id}")
        return jsonify(_entry_to_model(entry).model_dump())

    @app.get("/api/<env>/leaderboard.csv")
    def leaderboard_csv_route(env: str):
        from flask import Response
        from .csv_export import leaderboard_csv
        get_env(env)
        version = request.args.get("version", "v1")
        entries = app.extensions["backends"].leaderboard.list_entries(env, version)
        body = leaderboard_csv(entries)
        return Response(
            body, mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={env}_{version}_leaderboard.csv"},
        )

    @app.get("/api/<env>/leaderboard/<entry_id>.csv")
    def entry_csv_route(env: str, entry_id: str):
        from flask import Response
        from .csv_export import detailed_csv
        version = request.args.get("version", "v1")
        entry = app.extensions["backends"].leaderboard.get_entry(env, version, entry_id)
        if entry is None:
            raise KeyError(f"entry {entry_id}")
        body = detailed_csv(entry)
        agent = entry.get("agent_name", entry_id).replace(" ", "_")
        return Response(
            body, mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={env}_{agent}.csv"},
        )

    @app.get("/api/<env>/leaderboard/<entry_id>/plots/<kind>")
    def entry_plot_route(env: str, entry_id: str, kind: str):
        from flask import Response
        from .viz import per_partner_bars
        version = request.args.get("version", "v1")
        entry = app.extensions["backends"].leaderboard.get_entry(env, version, entry_id)
        if entry is None:
            raise KeyError(f"entry {entry_id}")
        if kind == "per_partner":
            png = per_partner_bars(entry, normalized=False)
        elif kind == "per_partner_normalized":
            png = per_partner_bars(entry, normalized=True)
        else:
            raise ValueError(f"unknown plot kind '{kind}'. supported: per_partner, per_partner_normalized")
        return Response(png, mimetype="image/png")

    @app.get("/api/<env>/leaderboard/plots/<kind>")
    def env_plot_route(env: str, kind: str):
        from flask import Response
        from .viz import comparison_bars, coverage_heatmap
        get_env(env)
        version = request.args.get("version", "v1")
        entries = app.extensions["backends"].leaderboard.list_entries(env, version)
        if kind == "comparison":
            png = comparison_bars(entries, metric="aggregate_score")
        elif kind == "coverage":
            png = coverage_heatmap(entries)
        else:
            raise ValueError(f"unknown plot kind '{kind}'. supported: comparison, coverage")
        return Response(png, mimetype="image/png")

    @app.get("/api/comparison.md")
    def cross_env_comparison_md():
        from flask import Response
        from .csv_export import comparison_markdown
        version = request.args.get("version", "v1")
        backends = app.extensions["backends"]
        entries_by_env: dict[str, list[dict]] = {}
        for env_name in env_registry():
            try:
                entries_by_env[env_name] = backends.leaderboard.list_entries(env_name, version)
            except Exception:
                continue
        body = comparison_markdown(entries_by_env)
        return Response(body, mimetype="text/markdown")

    # --------- static frontend passthrough ---------

    frontend_dist = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
    )
    if os.path.isdir(frontend_dist):
        @app.get("/")
        def _index():
            return send_from_directory(frontend_dist, "index.html")

        @app.get("/<path:path>")
        def _static(path: str):
            full = os.path.join(frontend_dist, path)
            if os.path.isfile(full):
                return send_from_directory(frontend_dist, path)
            return send_from_directory(frontend_dist, "index.html")


def _entry_to_model(entry: dict) -> LeaderboardEntry:
    """Disk schema → Pydantic. Tolerates missing optional fields."""
    return LeaderboardEntry(
        id=entry["id"],
        env=entry["env"],
        version=entry["version"],
        agent_name=entry["agent_name"],
        aggregate_score=float(entry.get("aggregate_score", 0.0)),
        aggregate=AggregateScore(**entry["aggregate"]),
        per_partner=[PerPartnerScore(**p) for p in entry.get("per_partner", [])],
        ego_kind=entry.get("ego_kind", "builtin"),
        builtin_key=entry.get("builtin_key"),
        checkpoint_sha256=entry.get("checkpoint_sha256"),
        num_episodes=int(entry.get("num_episodes", 0)),
        eval_seed=int(entry.get("eval_seed", 0)),
        notes=entry.get("notes"),
        created_at=float(entry.get("created_at", 0.0)),
        wall_clock_seconds=float(entry.get("wall_clock_seconds", 0.0)),
    )


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", 5174))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG") == "1")
