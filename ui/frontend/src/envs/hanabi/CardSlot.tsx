import type { HanabiBelief, HanabiCard } from "./types"
import { COLOR_BG, COLOR_DOT, COLOR_TEXT, rankGlyph } from "./cards"

// A single card slot in a hand. Three visual modes:
//   1. identity known (partner's hand, or post-play reveal) - colored + rank shown
//   2. identity unknown but hinted (human's own hand, after a hint) - partial info
//   3. identity unknown, unhinted (human's own hand, no hints) - blank slot
//
// We deliberately render the human's own hand from beliefs only.
// The partner hand renders from identity.

type Props = {
    card?: HanabiCard
    belief?: HanabiBelief
    slotIndex: number
    selected?: boolean
    onClick?: () => void
    isHumanHand?: boolean
    disabled?: boolean
}

export function CardSlot({ card, belief, slotIndex, selected, onClick, isHumanHand, disabled }: Props) {
    const isEmpty = !card && !belief

    let bgClass = "bg-ink-800/60 border-ink-700"
    let label: React.ReactNode = <span className="text-ink-500 text-xs">slot {slotIndex + 1}</span>
    let topLeft: React.ReactNode = null

    if (card?.known && card.color !== null && card.rank !== null) {
        bgClass = COLOR_BG[card.color]
        label = (
            <span className={`${COLOR_TEXT[card.color]} text-2xl font-bold`}>
                {rankGlyph(card.rank)}
            </span>
        )
    } else if (isHumanHand && belief) {
        const { color_hinted, rank_hinted, color, rank } = belief
        if (color_hinted && color !== null) {
            bgClass = COLOR_BG[color]
        }
        if (rank_hinted && rank !== null) {
            label = (
                <span className={`${color_hinted && color !== null ? COLOR_TEXT[color] : "text-ink-200"} text-2xl font-bold`}>
                    {rankGlyph(rank)}
                </span>
            )
        } else if (color_hinted && color !== null) {
            label = <span className={`${COLOR_TEXT[color]} text-xs font-medium`}>any</span>
        }
        const poss = belief.possible.length
        if (poss > 0 && poss < 25) {
            topLeft = (
                <span className="absolute top-1 left-1 text-[10px] text-ink-400 font-mono tabular-nums">
                    {poss}
                </span>
            )
        }
    }

    const clickable = onClick && !disabled && !isEmpty
    return (
        <button
            type="button"
            onClick={clickable ? onClick : undefined}
            disabled={!clickable}
            className={`
                relative h-28 w-20 rounded-lg border-2 transition-all duration-150
                flex items-center justify-center select-none
                ${bgClass}
                ${selected ? "ring-2 ring-offset-2 ring-offset-ink-950 ring-white scale-[1.04]" : ""}
                ${clickable ? "hover:scale-[1.03] cursor-pointer" : "cursor-default"}
                ${disabled ? "opacity-50" : ""}
            `}
            aria-label={ariaLabel(card, belief, slotIndex, isHumanHand)}
        >
            {topLeft}
            {label}
            {isHumanHand && belief && (
                <div className="absolute bottom-1 right-1 flex gap-0.5">
                    {belief.color_hinted && belief.color !== null && (
                        <span className={`w-2 h-2 rounded-full ${COLOR_DOT[belief.color]}`} />
                    )}
                    {belief.rank_hinted && belief.rank !== null && (
                        <span className="text-[10px] text-ink-200 font-mono font-bold">
                            {rankGlyph(belief.rank)}
                        </span>
                    )}
                </div>
            )}
        </button>
    )
}

function ariaLabel(card?: HanabiCard, belief?: HanabiBelief, idx: number = 0, isHuman?: boolean): string {
    if (isHuman && belief) {
        const c = belief.color_hinted && belief.color !== null ? `color ${belief.color}` : "unknown color"
        const r = belief.rank_hinted && belief.rank !== null ? `rank ${rankGlyph(belief.rank)}` : "unknown rank"
        return `your slot ${idx + 1}: ${c}, ${r}`
    }
    if (card?.known && card.color !== null && card.rank !== null) {
        return `partner slot ${idx + 1}: ${["red", "yellow", "green", "white", "blue"][card.color]} ${rankGlyph(card.rank)}`
    }
    return `slot ${idx + 1}: empty`
}
