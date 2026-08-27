import type { HanabiCard } from "./types"
import { COLOR_BG, COLOR_TEXT, rankGlyph } from "./cards"

// Compact grid of discarded cards, grouped by color column.
// Hidden by default; expand on click for full history.

type Props = {
    discards: HanabiCard[]
    numColors: number
    numRanks: number
}

export function DiscardPile({ discards, numColors, numRanks }: Props) {
    // Bucket by (color, rank) then render each color column.
    const buckets: Record<string, number> = {}
    for (const c of discards) {
        if (c.color === null || c.rank === null) continue
        const key = `${c.color}-${c.rank}`
        buckets[key] = (buckets[key] ?? 0) + 1
    }

    return (
        <div className="flex gap-2 items-end">
            {Array.from({ length: numColors }, (_, color) => (
                <div key={color} className="flex flex-col-reverse gap-0.5">
                    {Array.from({ length: numRanks }, (_, rank) => {
                        const count = buckets[`${color}-${rank}`] ?? 0
                        if (count === 0) {
                            return (
                                <div
                                    key={rank}
                                    className="w-6 h-4 rounded border border-ink-800 bg-ink-900/40"
                                />
                            )
                        }
                        return (
                            <div
                                key={rank}
                                className={`
                                    relative w-6 h-4 rounded border-2 flex items-center justify-center
                                    text-[10px] font-bold tabular-nums
                                    ${COLOR_BG[color]} ${COLOR_TEXT[color]}
                                `}
                            >
                                {rankGlyph(rank)}
                                {count > 1 && (
                                    <span className="absolute -top-1 -right-1 bg-ink-950 border border-ink-700 rounded-full px-1 text-[8px] text-ink-300">
                                        {count}
                                    </span>
                                )}
                            </div>
                        )
                    })}
                </div>
            ))}
        </div>
    )
}
