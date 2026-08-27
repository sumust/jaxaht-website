type Props = {
    completed: number
    total: number
    current?: string | null
    compact?: boolean
}

export function ProgressBar({ completed, total, current, compact }: Props) {
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0
    return (
        <div className={compact ? "space-y-1" : "space-y-2"}>
            <div className="flex justify-between text-xs text-ink-400">
                <span>
                    {current ? `Evaluating vs ${current}` : "Starting…"}
                </span>
                <span className="font-mono tabular-nums">
                    {completed} / {total > 0 ? total : "?"}
                </span>
            </div>
            <div className="h-2 bg-ink-900 rounded-full overflow-hidden border border-ink-800">
                <div
                    className="h-full bg-hanabi-blue transition-all duration-300 ease-out"
                    style={{ width: `${pct}%` }}
                />
            </div>
        </div>
    )
}
