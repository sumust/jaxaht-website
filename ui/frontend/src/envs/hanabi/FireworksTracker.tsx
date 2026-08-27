import { COLOR_BG, COLOR_DOT, COLOR_TEXT, rankGlyph } from "./cards"

// One column per color, showing the current fireworks height (0-5).
// Filled pips = played cards; dim pip with "next" = the playable rank.

type Props = {
    fireworks: number[]
    numRanks: number
    colorNames: string[]
}

export function FireworksTracker({ fireworks, numRanks, colorNames }: Props) {
    return (
        <div className="flex gap-3 items-end justify-center">
            {fireworks.map((height, color) => (
                <div key={color} className="flex flex-col items-center gap-1">
                    <div className="text-xs text-ink-400 font-mono uppercase tracking-wider">
                        {colorNames[color]?.[0] ?? "?"}
                    </div>
                    <div className="flex flex-col-reverse gap-0.5">
                        {Array.from({ length: numRanks }, (_, rankIdx) => {
                            const played = rankIdx < height
                            const isNext = rankIdx === height
                            return (
                                <div
                                    key={rankIdx}
                                    className={`
                                        h-6 w-8 rounded border flex items-center justify-center
                                        text-xs font-bold tabular-nums transition-all
                                        ${played ? COLOR_BG[color] : ""}
                                        ${played ? COLOR_TEXT[color] : "border-ink-700 text-ink-600 bg-ink-900/40"}
                                        ${isNext ? "ring-1 ring-ink-500 ring-dashed" : ""}
                                    `}
                                >
                                    {played ? rankGlyph(rankIdx) : isNext ? "·" : ""}
                                </div>
                            )
                        })}
                    </div>
                    <div className={`w-2 h-2 rounded-full ${COLOR_DOT[color]}`} />
                </div>
            ))}
        </div>
    )
}
