"""Multi-game study sessions for Prolific workers.

Wraps N sequential LiveSessions (one per game) with warmup + real phases,
Prolific metadata, and a completion code generated on finish. Generic
across envs: each EnvRenderer supplies its own StudyConfig.
"""
from __future__ import annotations

import random
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .envs import EnvRenderer, get as get_env
from .sessions import LiveSession, SessionManager


@dataclass
class ProlificMeta:
    """Prolific URL params attached to a study session for trajectory metadata."""
    prolific_pid: Optional[str] = None
    study_id: Optional[str] = None
    prolific_session_id: Optional[str] = None
    data_source: str = ""


@dataclass
class StudyConfig:
    """Per-env study setup: how many games of each kind + per-game env_kwargs."""
    num_warmup: int
    num_real: int
    game_env_kwargs: list[dict]  # len == num_warmup + num_real
    warmup_partner_keys: Optional[list[str]] = None  # if None, picks randomly per game
    real_partner_keys: Optional[list[str]] = None    # if None, picks randomly per game

    def total_games(self) -> int:
        return self.num_warmup + self.num_real


@dataclass
class StudySession:
    """One Prolific worker's run through a study. Holds a list of game
    session_ids, advances on completion, generates a completion code."""
    study_id: str
    env_name: str
    config: StudyConfig
    prolific: ProlificMeta
    current_game_index: int = 0
    game_session_ids: list[Optional[str]] = field(default_factory=list)
    completion_code: Optional[str] = None
    session_complete: bool = False
    last_seen_at: float = field(default_factory=time.time)

    def is_warmup(self) -> bool:
        return self.current_game_index < self.config.num_warmup

    def current_game_id(self) -> Optional[str]:
        if 0 <= self.current_game_index < len(self.game_session_ids):
            return self.game_session_ids[self.current_game_index]
        return None


class StudySessionManager:
    def __init__(self, session_manager: SessionManager):
        self._sm = session_manager
        self._lock = threading.Lock()
        self._active: dict[str, StudySession] = {}

    def start(self, env_name: str, prolific: ProlificMeta,
              config: Optional[StudyConfig] = None) -> StudySession:
        renderer = get_env(env_name)
        cfg = config or renderer.study_config()
        if len(cfg.game_env_kwargs) != cfg.total_games():
            raise ValueError(
                f"study config: {len(cfg.game_env_kwargs)} game_env_kwargs "
                f"!= {cfg.total_games()} total games"
            )

        study_id = _gen_id("study_")
        study = StudySession(
            study_id=study_id,
            env_name=env_name,
            config=cfg,
            prolific=prolific,
            game_session_ids=[None] * cfg.total_games(),
        )
        # Kick off game 0 immediately so the client gets a state to render.
        first_sid = self._launch_game(renderer, study, 0)
        study.game_session_ids[0] = first_sid

        with self._lock:
            self._active[study_id] = study
        return study

    def get(self, study_id: str) -> StudySession:
        with self._lock:
            s = self._active.get(study_id)
        if s is None:
            raise KeyError(f"study {study_id} expired or unknown")
        s.last_seen_at = time.time()
        return s

    def step(self, study_id: str, ui_action: dict) -> dict:
        """Step current game (mirrors /play/step's full flow: human action +
        partner action if their turn next). On game-done, advance to next;
        on study-done, generate completion code."""
        study = self.get(study_id)
        renderer = get_env(study.env_name)
        cur_sid = study.current_game_id()
        if cur_sid is None:
            raise RuntimeError(f"study {study_id}: no current game")
        live = self._sm.get(cur_sid)

        from .helpers import current_player as _current_player, next_rng
        human_idx = renderer.HUMAN_AGENT_IDX
        partner_idx = 1 - human_idx

        # Same shape as /play/step: human action, then partner if next.
        action = renderer.action_from_ui(ui_action, live.state)
        state_before_human = live.state
        obs, state, reward, done, info = renderer.step(
            live.env, live.state, {human_idx: action}, next_rng(),
        )
        live.state = state
        human_event = renderer.describe_action(
            action, state_before_human, live.state, float(reward), human_idx,
        )
        events: list[dict] = [human_event]
        live.history.append(human_event)

        partner_acted = False
        if not done:
            cur_player = _current_player(live.state, human_idx)
            if cur_player == partner_idx:
                partner_action = int(live.partner.get_action(obs, live.state, next_rng()))
                state_before_partner = live.state
                obs, state, p_reward, done, info = renderer.step(
                    live.env, live.state, {partner_idx: partner_action}, next_rng(),
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

        advanced = False
        prev_idx = study.current_game_index
        if done:
            # Stamp end_time on the game that just finished (for per-game duration metric).
            live.end_time = time.time()
            if study.current_game_index < study.config.total_games() - 1:
                study.current_game_index += 1
                next_sid = self._launch_game(renderer, study, study.current_game_index)
                study.game_session_ids[study.current_game_index] = next_sid
                advanced = True
            else:
                study.session_complete = True
                study.completion_code = _gen_completion_code()

        return {
            "study_id": study.study_id,
            "current_game_index": study.current_game_index,
            "total_games": study.config.total_games(),
            "num_warmup": study.config.num_warmup,
            "is_warmup": study.is_warmup(),
            "session_complete": study.session_complete,
            "completion_code": study.completion_code,
            "game_just_advanced": advanced,
            "prev_game_index": prev_idx if advanced else None,
            "reward": float(reward),
            "done": bool(done),
            "partner_acted": partner_acted,
            "events": events,
        }

    def save_all(self, study_id: str, agent_name: str = "Anonymous") -> list[dict]:
        """Build rich trajectory dicts for every completed game in this study.
        Caller (route handler) persists them via TrajectoryStore.

        Per-trajectory metadata mirrors Johnny's human_data_collecting save_episode:
        timing (start_time/end_time/duration), env_kwargs (grid_size/num_food/...),
        partner identity, warmup flag, sequence-aware kind tag for folder split.
        """
        study = self.get(study_id)
        trajs: list[dict] = []
        for idx, sid in enumerate(study.game_session_ids):
            if sid is None:
                continue
            try:
                live = self._sm.get(sid)
            except KeyError:
                continue
            is_warmup = idx < study.config.num_warmup
            env_kwargs = (
                study.config.game_env_kwargs[idx]
                if idx < len(study.config.game_env_kwargs) else {}
            )
            start = getattr(live, "start_time", None)
            end = getattr(live, "end_time", None) or time.time()
            traj = {
                "env": study.env_name,
                "agent_name": agent_name,
                "partner_key": live.partner_key,
                "score": live.renderer.score_summary(live.state, {}),
                "history": live.history,
                "prolific": {
                    "prolific_pid": study.prolific.prolific_pid,
                    "study_id": study.prolific.study_id,
                    "prolific_session_id": study.prolific.prolific_session_id,
                    "data_source": study.prolific.data_source,
                },
                "study_id": study.study_id,
                "game_index": idx,
                "is_warmup": is_warmup,
                "kind": "warmup" if is_warmup else "real",
                "env_kwargs": env_kwargs,
                "start_time": start,
                "end_time": end,
                "duration": (end - start) if (start and end) else None,
                "total_steps": len(live.history),
                "completion_code": study.completion_code,
            }
            trajs.append(traj)
        return trajs

    def _launch_game(self, renderer: EnvRenderer, study: StudySession, idx: int) -> str:
        env_kwargs = study.config.game_env_kwargs[idx]
        partner_key = self._pick_partner(renderer, study, idx)
        # SessionManager.start() handles the env+partner+state setup.
        live, _obs = self._sm.start(
            env_name=study.env_name,
            partner_key=partner_key,
            env_kwargs=env_kwargs,
            seed=None,
        )
        # Stamp start_time so save_all can record per-game duration.
        live.start_time = time.time()
        return live.session_id

    def _pick_partner(self, renderer: EnvRenderer, study: StudySession, idx: int) -> str:
        cfg = study.config
        is_warmup = idx < cfg.num_warmup
        pool = (cfg.warmup_partner_keys if is_warmup else cfg.real_partner_keys)
        if pool:
            return random.choice(pool)
        # Fallback: pick from full available_partners list (Johnny's _choose_agent equivalent).
        partners = [p.key for p in renderer.available_partners()]
        return random.choice(partners) if partners else renderer.default_partner_key()

    def purge_stale(self, ttl_seconds: int = 7200) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            stale = [sid for sid, s in self._active.items() if s.last_seen_at < cutoff]
            for sid in stale:
                self._active.pop(sid)
        return len(stale)


def _gen_id(prefix: str = "study_") -> str:
    import uuid
    return prefix + uuid.uuid4().hex[:12]


def _gen_completion_code(length: int = 8) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))
