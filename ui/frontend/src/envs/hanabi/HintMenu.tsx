import type { HanabiAction, HanabiState } from "./types"
import { COLOR_BG, COLOR_TEXT, rankGlyph } from "./cards"

// Modal-ish panel that appears when a partner slot is selected.
// Hints are card-attribute (color/rank) based in Hanabi, not slot-based,
// so we offer the color + rank that the targeted card has, plus any
// other color/rank the partner holds (so the user can pick any legal hint).

type Props = {
    state: HanabiState
    partnerSlot: number
    onHint: (a: HanabiAction) => void
    onCancel: () => void
}

export function HintMenu({ state, partnerSlot, onHint, onCancel }: Props) {
    const selectedCard = state.partner_hand[partnerSlot]

    // Collect all colors + ranks held in partner hand (those are the
    // only legal hints to give).
    const colorsHeld = new Set<number>()
    const ranksHeld = new Set<number>()
    for (const card of state.partner_hand) {
        if (card.color !== null) colorsHeld.add(card.color)
        if (card.rank !== null) ranksHeld.add(card.rank)
    }

    return (
        <div className="col-span-12 card-surface p-6 animate-slide-up">
            <div className="flex items-start justify-between mb-4">
                <div>
                    <h4 className="text-sm font-medium text-ink-200">
                        Give a hint
                    </h4>
                    <p className="text-xs text-ink-500 mt-1">
                        Targeting partner slot {partnerSlot + 1}
                        {selectedCard?.known && selectedCard.color !== null && selectedCard.rank !== null && (
                            <>
                                {" "}({state.color_names[selectedCard.color]} {rankGlyph(selectedCard.rank)})
                            </>
                        )}
                        {" - "}costs 1 info token
                    </p>
                </div>
                <button className="btn-ghost" onClick={onCancel}>
                    Cancel
                </button>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div>
                    <div className="text-xs uppercase tracking-wider text-ink-500 mb-2">
                        Color hints
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {Array.from({ length: state.num_colors }, (_, c) => {
                            const held = colorsHeld.has(c)
                            return (
                                <button
                                    key={c}
                                    onClick={() => onHint({ type: "hint_color", color: c })}
                                    disabled={!held}
                                    className={`
                                        px-3 py-2 rounded-lg border-2 text-sm font-medium
                                        transition-all
                                        ${held ? `${COLOR_BG[c]} ${COLOR_TEXT[c]} hover:scale-105` : "bg-ink-900/40 border-ink-800 text-ink-600 cursor-not-allowed"}
                                    `}
                                >
                                    {state.color_names[c]}
                                </button>
                            )
                        })}
                    </div>
                </div>

                <div>
                    <div className="text-xs uppercase tracking-wider text-ink-500 mb-2">
                        Rank hints
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {Array.from({ length: state.num_ranks }, (_, r) => {
                            const held = ranksHeld.has(r)
                            return (
                                <button
                                    key={r}
                                    onClick={() => onHint({ type: "hint_rank", rank: r })}
                                    disabled={!held}
                                    className={`
                                        px-3 py-2 rounded-lg border-2 text-sm font-bold tabular-nums
                                        transition-all
                                        ${held ? "bg-ink-800 border-ink-600 text-ink-100 hover:bg-ink-700" : "bg-ink-900/40 border-ink-800 text-ink-600 cursor-not-allowed"}
                                    `}
                                >
                                    {rankGlyph(r)}
                                </button>
                            )
                        })}
                    </div>
                </div>
            </div>

            <p className="text-xs text-ink-500 mt-4">
                A hint reveals every card in partner's hand that matches the attribute, not just slot {partnerSlot + 1}.
            </p>
        </div>
    )
}
