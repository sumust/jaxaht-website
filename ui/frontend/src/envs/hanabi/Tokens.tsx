// Info-token + life-token row.
// Info tokens are the hints you can give. Life tokens are bombs.

type Props = {
    label: string
    current: number
    max: number
    variant: "info" | "life"
}

export function TokenRow({ label, current, max, variant }: Props) {
    const pipColor = variant === "info" ? "bg-hanabi-blue" : "bg-hanabi-red"
    return (
        <div className="flex items-center gap-3">
            <span className="text-xs uppercase tracking-wider text-ink-500 font-medium w-10">
                {label}
            </span>
            <div className="flex gap-1">
                {Array.from({ length: max }).map((_, i) => {
                    const active = i < current
                    return (
                        <span
                            key={i}
                            className={`
                                w-4 h-4 rounded-full transition-all
                                ${active ? pipColor : "bg-ink-800 border border-ink-700"}
                                ${active ? "shadow-sm" : ""}
                            `}
                        />
                    )
                })}
            </div>
            <span className="ml-2 text-sm font-mono text-ink-300 tabular-nums">
                {current}/{max}
            </span>
        </div>
    )
}
