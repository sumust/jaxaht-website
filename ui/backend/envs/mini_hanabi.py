from __future__ import annotations

from .hanabi import HanabiRenderer, _MINI_KWARGS


class MiniHanabiRenderer(HanabiRenderer):
    ENV_NAME = "mini-hanabi"
    DISPLAY_NAME = "Mini-Hanabi"
    DEFAULT_KWARGS = _MINI_KWARGS
    OVERVIEW = "Smaller Hanabi (3 colors, 3 ranks, 3-card hand). Max score 9."
    STATS = {
        "players": "2",
        "actions": "13",
        "max score": "9",
    }
