"""Environment adapter contract.

Adding a new environment to ui is a matter of subclassing
EnvRenderer and implementing ~10 methods. Routes in ``app.py`` stay
env-agnostic and dispatch via the registry.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PartnerSpec:
    """One held-out partner the user can pick to play against."""
    key: str
    display_name: str
    description: str
    load_fn: Callable[[], Any]   # lazy, called on first use
    difficulty: str = ""    # deprecated; no longer shown in UI
    tags: list[str] = field(default_factory=list)


@dataclass
class BuiltinEgoSpec:
    """One pre-registered ego the user can submit without uploading.

    Useful for baseline entries (random / self-play / published
    checkpoints) and for making the Submit flow testable without an
    actual file upload.
    """
    key: str
    display_name: str
    description: str
    load_fn: Callable[[], Callable]   # () -> (obs, state, rng) -> action
    tags: list[str] = field(default_factory=list)


@dataclass
class ModeSpec:
    """A frontend mode offered by an env. Shown on the home page."""
    key: str                # "play" | "submit" | "leaderboard" | "replay"
    display_name: str
    enabled: bool = True


class EnvRenderer(ABC):
    """Per-environment adapter. All env-specific logic lives here.

    Subclasses must set the identity constants and implement the
    abstract methods. Routes never import env modules directly; they go
    through the registry keyed on ENV_NAME.
    """

    # Class-level identity. Set on subclass.
    ENV_NAME: str = ""
    DISPLAY_NAME: str = ""
    HUMAN_AGENT_IDX: int = 0
    DEFAULT_KWARGS: dict[str, Any] = {}
    # Turn-based: only the current_player acts each step (Hanabi).
    # Simultaneous: both agents act every step (LBF, Overcooked).
    IS_TURN_BASED: bool = False

    # Presentation hints for the Home-page card.
    OVERVIEW: str = ""              # one-sentence tagline
    STATS: dict[str, str] = {}      # quick facts shown under the title
    ACCENT: str = "blue"            # frontend chooses a color

    # Visibility. HIDDEN envs don't show on the Home page (used for
    # mock/dev harnesses). READY=False envs render as "coming soon"
    # placeholders with disabled action buttons.
    HIDDEN: bool = False
    READY: bool = True

    # ------- env lifecycle -------

    @abstractmethod
    def make_env(self, kwargs: dict | None = None):
        """Build an env instance. kwargs overrides DEFAULT_KWARGS per session."""

    @abstractmethod
    def reset(self, env, rng):
        """Return (obs, state) for a fresh episode."""

    @abstractmethod
    def step(self, env, state, actions, rng):
        """Advance one timestep. Return (obs, state, reward, done, info)."""

    # ------- partner catalog -------

    @abstractmethod
    def available_partners(self) -> list[PartnerSpec]:
        """List of partners the UI lets the human pick."""

    # ------- UI contract -------

    @abstractmethod
    def serialize_state(self, state, obs) -> dict:
        """Convert env state to a JSON-serializable payload the
        frontend renders. Shape is env-specific; the frontend's env
        adapter knows how to read it.
        """

    @abstractmethod
    def action_from_ui(self, ui_payload: dict, state) -> int:
        """Map a UI action description (button click, keystroke, drag
        target) to the env action index. Raises ValueError on an
        illegal action.
        """

    @abstractmethod
    def score_summary(self, state, info) -> dict:
        """Env-specific scoreboard fields: lives + fireworks for
        Hanabi, fruits_collected for LBF, etc. Rendered in the shared
        ScoreBoard component.
        """

    # ------- optional hooks -------

    def builtin_egos(self) -> list[BuiltinEgoSpec]:
        """Pre-registered egos the user can submit without uploading.

        Default: empty list. Override to provide baselines / published
        checkpoints so the leaderboard has reference entries out of the
        box.
        """
        return []

    def describe_action(
        self,
        action: int,
        state_before,
        state_after,
        reward: float,
        player: int,
    ) -> dict:
        """Return a human-readable event dict describing one action.

        Default: bare action index. Override to produce richer
        'played red 3 -> bombed' events the frontend GameLog can show.
        """
        return {"player": player, "action": int(action), "reward": float(reward)}

    def keyboard_controls(self) -> dict[str, int]:
        """Map keystroke to action index. Empty dict = mouse-only UI."""
        return {}

    def modes(self) -> list[ModeSpec]:
        """Modes offered. Override to disable modes per env (e.g. Hanabi
        supports all four, a toy env might only support play)."""
        return [
            ModeSpec("play", "Play with an AI"),
            ModeSpec("submit", "Submit a checkpoint"),
            ModeSpec("leaderboard", "Leaderboard"),
            ModeSpec("demo", "Watch two policies"),
        ]

    def default_partner_key(self) -> str:
        """Which partner should be pre-selected in the UI. Defaults to
        the first in ``available_partners``."""
        partners = self.available_partners()
        if not partners:
            raise ValueError(f"{self.ENV_NAME}: no partners registered")
        return partners[0].key

    def study_config(self):
        """Default StudyConfig for Prolific multi-game flows.

        Default: 1 warmup + 3 real games, all with DEFAULT_KWARGS. Override
        per env to vary configs per game or pin specific partner pools.
        Returns a study_session.StudyConfig.
        """
        from ..study_session import StudyConfig
        num_warmup, num_real = 1, 3
        total = num_warmup + num_real
        return StudyConfig(
            num_warmup=num_warmup,
            num_real=num_real,
            game_env_kwargs=[dict(self.DEFAULT_KWARGS) for _ in range(total)],
        )
