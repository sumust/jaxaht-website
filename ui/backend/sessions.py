"""Live-game session manager.

Session metadata (id, env, partner_key, config) persists via the
SessionStore. The actual env/state/partner pytrees live in-process in
_ACTIVE because jax pytrees don't pickle cleanly. If the server
restarts, sessions expire; clients resume by starting a new one.

Routes should go through ``SessionManager`` rather than poking
``_ACTIVE`` or the store directly.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .envs import EnvRenderer, get as get_env
from .helpers import next_rng as _next_rng
from .storage import SessionStore

log = logging.getLogger(__name__)


@dataclass
class LiveSession:
    """Everything attached to a running game. Stored in memory."""
    session_id: str
    env_name: str
    renderer: EnvRenderer
    env: Any
    state: Any
    partner: Any
    partner_key: str
    last_seen_at: float
    history: list[dict]
    last_obs: Any = None


class SessionManager:
    def __init__(self, store: SessionStore):
        self.store = store
        self._lock = threading.Lock()
        self._active: dict[str, LiveSession] = {}

    def start(self, env_name: str, partner_key: str, env_kwargs: dict, seed: int | None):
        renderer = get_env(env_name)
        partners = {p.key: p for p in renderer.available_partners()}
        if partner_key not in partners:
            raise ValueError(f"unknown partner '{partner_key}' for env '{env_name}'")

        env = renderer.make_env(env_kwargs)
        rng = _make_rng_for_seed(seed)
        obs, state = renderer.reset(env, rng)
        partner = partners[partner_key].load_fn()

        sid = self.store.create(env_name, {
            "partner_key": partner_key,
            "env_kwargs": env_kwargs,
            "seed": seed,
        })
        live = LiveSession(
            session_id=sid,
            env_name=env_name,
            renderer=renderer,
            env=env,
            state=state,
            partner=partner,
            partner_key=partner_key,
            last_seen_at=time.time(),
            history=[],
            last_obs=obs,
        )

        # If partner is current_player at reset (env chose to go second
        # for the human), let the partner act before returning control
        # so the UI never sits on "not your turn" with no way to advance.
        obs = self._advance_partner_turns(live, obs)
        live.last_obs = obs

        with self._lock:
            self._active[sid] = live
        return live, obs

    def _advance_partner_turns(self, live: "LiveSession", obs):
        """If current_player is the partner, run partner actions until
        either the human's turn or the game ends. Keeps the initial
        GET-new-game payload in a state the UI can render + act on."""
        from .helpers import current_player, next_rng
        renderer = live.renderer
        human_idx = renderer.HUMAN_AGENT_IDX
        partner_idx = 1 - human_idx
        # Bounded loop so a misbehaving partner can't hang the route.
        for _ in range(8):
            current = current_player(live.state, human_idx)
            if current != partner_idx:
                return obs
            partner_action = int(
                live.partner.get_action(obs, live.state, next_rng())
            )
            state_before = live.state
            obs, state, p_reward, done, _info = renderer.step(
                live.env,
                live.state,
                {partner_idx: partner_action},
                next_rng(),
            )
            live.state = state
            live.history.append(renderer.describe_action(
                partner_action, state_before, live.state,
                float(p_reward), partner_idx,
            ))
            if done:
                return obs
        return obs

    def get(self, session_id: str) -> LiveSession:
        with self._lock:
            live = self._active.get(session_id)
        if live is None:
            raise KeyError(f"session {session_id} expired or unknown")
        live.last_seen_at = time.time()
        return live

    def end(self, session_id: str) -> None:
        with self._lock:
            self._active.pop(session_id, None)
        self.store.delete(session_id)

    def purge_stale(self, ttl_seconds: int = 3600) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            stale = [sid for sid, s in self._active.items() if s.last_seen_at < cutoff]
            for sid in stale:
                self._active.pop(sid)
        for sid in stale:
            self.store.delete(sid)
        return len(stale)

    def start_purge_thread(self, ttl_seconds: int = 3600, interval_seconds: int = 300) -> None:
        """Fire a background daemon thread that calls purge_stale every
        interval_seconds. No-op if already started."""
        if getattr(self, "_purge_started", False):
            return
        self._purge_started = True

        def _loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    purged = self.purge_stale(ttl_seconds)
                    if purged > 0:
                        log.info("session_purge purged=%d ttl_s=%d", purged, ttl_seconds)
                except Exception as exc:  # noqa: BLE001
                    log.exception("session purge crashed: %s", exc)

        t = threading.Thread(target=_loop, daemon=True, name="session-purge")
        t.start()


def _make_rng_for_seed(seed: int | None):
    """Build an RNG from an explicit seed if given, else fall back to
    next_rng()'s time-based seed. Separate from next_rng so sessions
    can be reproducible when the caller provides a seed."""
    if seed is None:
        return _next_rng()
    try:
        import jax  # noqa: WPS433
        return jax.random.PRNGKey(seed)
    except ImportError:
        import random as _rng
        return _rng.Random(seed)
