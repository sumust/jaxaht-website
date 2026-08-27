// Tailwind color + glyph helpers for Hanabi cards.
// Keeping this centralized so the CSS classes stay consistent across
// CardSlot, FireworksTracker, DiscardPile, and HintHistory.

export const COLOR_BG = [
    "bg-hanabi-red/20    border-hanabi-red",
    "bg-hanabi-yellow/20 border-hanabi-yellow",
    "bg-hanabi-green/20  border-hanabi-green",
    "bg-ink-200/10       border-ink-200",   // white - can't be purely white on dark bg
    "bg-hanabi-blue/20   border-hanabi-blue",
]

export const COLOR_TEXT = [
    "text-hanabi-red",
    "text-hanabi-yellow",
    "text-hanabi-green",
    "text-ink-100",
    "text-hanabi-blue",
]

export const COLOR_DOT = [
    "bg-hanabi-red",
    "bg-hanabi-yellow",
    "bg-hanabi-green",
    "bg-ink-200",
    "bg-hanabi-blue",
]

export const COLOR_LABEL = ["R", "Y", "G", "W", "B"]

export function rankGlyph(rank: number): string {
    return ["1", "2", "3", "4", "5"][rank] ?? "?"
}
