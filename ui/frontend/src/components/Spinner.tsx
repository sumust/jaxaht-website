// Tiny spinner. CSS-only, no deps, respects reduced-motion.

type Props = {
    size?: "sm" | "md" | "lg"
    label?: string
}

const SIZE_PX = { sm: 14, md: 20, lg: 32 } as const

export function Spinner({ size = "md", label }: Props) {
    const px = SIZE_PX[size]
    return (
        <span className="inline-flex items-center gap-2" role="status" aria-live="polite">
            <span
                className="inline-block rounded-full border-2 border-ink-700 border-t-ink-200 motion-safe:animate-spin motion-reduce:animate-pulse"
                style={{ width: px, height: px }}
                aria-hidden="true"
            />
            {label && <span className="text-sm text-ink-400">{label}</span>}
        </span>
    )
}
