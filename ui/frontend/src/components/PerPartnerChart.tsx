// Horizontal-bar chart of per-partner scores for one leaderboard entry.
// CI whiskers inline; normalized-score scale 0-1 on x-axis.

import type { PerPartnerScore } from "../api/client"

type Props = {
    scores: PerPartnerScore[]
    mode?: "normalized" | "raw"
}

export function PerPartnerChart({ scores, mode = "normalized" }: Props) {
    if (scores.length === 0) {
        return <div className="text-xs text-ink-500">No partners evaluated.</div>
    }

    const pick = (s: PerPartnerScore) => mode === "normalized"
        ? { v: s.normalized_mean, lo: s.normalized_ci_low, hi: s.normalized_ci_high }
        : { v: s.mean, lo: s.ci_low, hi: s.ci_high }

    const vals = scores.map(pick)
    const maxHi = Math.max(...vals.map((v) => v.hi), 1)
    const minLo = Math.min(...vals.map((v) => v.lo), 0)
    const span = Math.max(1e-9, maxHi - minLo)

    return (
        <div className="space-y-1.5 text-xs">
            {scores.map((s, i) => {
                const { v, lo, hi } = vals[i]
                const leftPct = ((lo - minLo) / span) * 100
                const widthPct = ((hi - lo) / span) * 100
                const markerPct = ((v - minLo) / span) * 100
                return (
                    <div key={s.key} className="grid grid-cols-[140px_1fr_70px] gap-2 items-center">
                        <div className="truncate text-ink-300" title={s.display_name}>
                            {s.display_name}
                        </div>
                        <div className="relative h-4 bg-ink-900 rounded border border-ink-800">
                            <div
                                className="absolute top-0.5 bottom-0.5 bg-hanabi-blue/30 rounded-sm"
                                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                                title={`CI [${lo.toFixed(2)}, ${hi.toFixed(2)}]`}
                            />
                            <div
                                className="absolute top-0 bottom-0 w-0.5 bg-hanabi-blue"
                                style={{ left: `${markerPct}%` }}
                            />
                        </div>
                        <div className="text-right font-mono tabular-nums text-ink-200">
                            {v.toFixed(3)}
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
