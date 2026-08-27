type Props = {
    score: Record<string, unknown>
}

// Env-agnostic score display. Renders whatever fields the backend
// adapter put in score_summary. Uses a 2-col grid and renders arrays
// as dot-rows, numbers as tabular-nums, strings as-is.

export function ScoreBoard({ score }: Props) {
    const entries = Object.entries(score)
    if (entries.length === 0) {
        return null
    }
    return (
        <div className="card-surface p-4 grid grid-cols-2 gap-x-6 gap-y-2">
            {entries.map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-wider text-ink-500">
                        {key.replace(/_/g, " ")}
                    </span>
                    <span className="font-mono text-sm text-ink-200 tabular-nums">
                        {renderValue(value)}
                    </span>
                </div>
            ))}
        </div>
    )
}

function renderValue(v: unknown): React.ReactNode {
    if (Array.isArray(v)) {
        return <span className="text-xs">[{v.join(", ")}]</span>
    }
    if (typeof v === "number") {
        return Number.isInteger(v) ? v.toString() : v.toFixed(2)
    }
    if (typeof v === "string") return v
    return String(v)
}
