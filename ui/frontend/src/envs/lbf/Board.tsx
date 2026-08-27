import { useEffect } from "react"
import type { LBFAction, LBFState } from "./types"

// LBF grid board. Agents are circles with level pips; food items are
// fruit-colored squares with level numbers; eaten food fades out.
//
// Keyboard: WASD + space + q. Arrow keys alias WASD.

type Props = {
    state: LBFState
    onAction: (action: LBFAction) => void
    disabled?: boolean
}

const AGENT_COLORS = ["bg-hanabi-blue", "bg-hanabi-yellow"]
const AGENT_BORDERS = ["border-hanabi-blue", "border-hanabi-yellow"]

export function LBFBoard({ state, onAction, disabled }: Props) {
    useEffect(() => {
        if (disabled) return
        const handler = (ev: KeyboardEvent) => {
            if (ev.target instanceof HTMLInputElement) return
            const key = ev.key.toLowerCase()
            const dirMap: Record<string, LBFAction["dir"]> = {
                w: "up", arrowup: "up",
                s: "down", arrowdown: "down",
                a: "left", arrowleft: "left",
                d: "right", arrowright: "right",
                " ": "load",
                q: "wait",
            }
            const dir = dirMap[key]
            if (dir) {
                ev.preventDefault()
                onAction({ type: "move", dir })
            }
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [disabled, onAction])

    const cellPx = Math.max(40, Math.min(80, 480 / state.grid_size))
    const boardPx = cellPx * state.grid_size

    // Lookup tables per cell for O(1) render
    const foodAt = new Map<string, LBFState["food"][number]>()
    for (const f of state.food) foodAt.set(`${f.x},${f.y}`, f)

    const agentAt = new Map<string, { agent: LBFState["agents"][number]; idx: number }>()
    state.agents.forEach((a, idx) => agentAt.set(`${a.x},${a.y}`, { agent: a, idx }))

    return (
        <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-9 space-y-4">
                <section className="card-surface p-8 flex items-center justify-center">
                    <div
                        className="relative border border-ink-800 rounded-lg overflow-hidden"
                        style={{ width: boardPx, height: boardPx, background: "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255,255,255,0.015) 1px, rgba(255,255,255,0.015) 2px)" }}
                    >
                        {Array.from({ length: state.grid_size }, (_, y) => (
                            Array.from({ length: state.grid_size }, (_, x) => {
                                const food = foodAt.get(`${x},${y}`)
                                const agent = agentAt.get(`${x},${y}`)
                                return (
                                    <div
                                        key={`${x}-${y}`}
                                        className="absolute flex items-center justify-center"
                                        style={{
                                            left: x * cellPx,
                                            top: y * cellPx,
                                            width: cellPx,
                                            height: cellPx,
                                        }}
                                    >
                                        {food && !food.eaten && (
                                            <FoodTile level={food.level} size={cellPx} />
                                        )}
                                        {food?.eaten && (
                                            <div className="text-ink-800 text-xs font-mono opacity-30">·</div>
                                        )}
                                        {agent && (
                                            <AgentChip
                                                idx={agent.idx}
                                                level={agent.agent.level}
                                                size={cellPx}
                                            />
                                        )}
                                    </div>
                                )
                            })
                        ))}
                    </div>
                </section>

                <ControlBar onAction={onAction} disabled={disabled} />
            </div>

            <aside className="col-span-12 lg:col-span-3 space-y-4">
                <section className="card-surface p-6">
                    <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-3">
                        Fruits
                    </h3>
                    <div className="space-y-2">
                        {state.food.map((f, i) => (
                            <div key={i} className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                    <FoodTile level={f.level} size={20} />
                                    <span className="text-ink-400 font-mono text-xs">
                                        ({f.x},{f.y})
                                    </span>
                                </div>
                                {f.eaten ? (
                                    <span className="text-xs text-hanabi-green">eaten</span>
                                ) : (
                                    <span className="text-xs text-ink-500">lvl {f.level}</span>
                                )}
                            </div>
                        ))}
                    </div>
                </section>

                <section className="card-surface p-6">
                    <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-3">
                        Agents
                    </h3>
                    <div className="space-y-2">
                        {state.agents.map((a, i) => (
                            <div key={i} className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                    <AgentChip idx={i} level={a.level} size={20} />
                                    <span className={`font-medium ${i === 0 ? "text-hanabi-blue" : "text-hanabi-yellow"}`}>
                                        {i === 0 ? "you" : "partner"}
                                    </span>
                                </div>
                                <span className="text-xs text-ink-500 font-mono">
                                    ({a.x},{a.y})
                                </span>
                            </div>
                        ))}
                    </div>
                </section>
            </aside>

            <div className="col-span-12 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-500">
                <Kbd keys={["W", "A", "S", "D"]} label="move" />
                <Kbd keys={["Space"]} label="load/collect" />
                <Kbd keys={["Q"]} label="wait" />
            </div>
        </div>
    )
}

function FoodTile({ level, size }: { level: number; size: number }) {
    return (
        <div
            className="rounded-md bg-hanabi-red/30 border-2 border-hanabi-red flex items-center justify-center text-hanabi-red font-bold"
            style={{
                width: size * 0.7,
                height: size * 0.7,
                fontSize: size * 0.3,
            }}
        >
            {level}
        </div>
    )
}

function AgentChip({ idx, level, size }: { idx: number; level: number; size: number }) {
    const color = AGENT_COLORS[idx % AGENT_COLORS.length]
    const border = AGENT_BORDERS[idx % AGENT_BORDERS.length]
    return (
        <div
            className={`rounded-full ${color}/30 border-2 ${border} flex items-center justify-center font-bold text-ink-100 shadow-lg`}
            style={{
                width: size * 0.5,
                height: size * 0.5,
                fontSize: size * 0.22,
            }}
        >
            {level}
        </div>
    )
}

function ControlBar({ onAction, disabled }: { onAction: (a: LBFAction) => void; disabled?: boolean }) {
    const btn = (dir: LBFAction["dir"], label: string) => (
        <button
            className="btn-outline !px-3 !py-2 min-w-0"
            disabled={disabled}
            onClick={() => onAction({ type: "move", dir })}
        >
            {label}
        </button>
    )
    return (
        <section className="card-surface p-4 flex items-center justify-center gap-2 flex-wrap">
            <div className="grid grid-cols-3 gap-1">
                <div />
                {btn("up", "↑")}
                <div />
                {btn("left", "←")}
                {btn("wait", "·")}
                {btn("right", "→")}
                <div />
                {btn("down", "↓")}
                <div />
            </div>
            <div className="w-px h-20 bg-ink-800 mx-2" />
            <button className="btn-primary h-20 !px-6" disabled={disabled} onClick={() => onAction({ type: "move", dir: "load" })}>
                Load
            </button>
        </section>
    )
}

function Kbd({ keys, label }: { keys: string[]; label: string }) {
    return (
        <span className="inline-flex items-center gap-1.5">
            {keys.map((k) => (
                <kbd
                    key={k}
                    className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-ink-700 bg-ink-900 text-ink-400"
                >
                    {k}
                </kbd>
            ))}
            <span>{label}</span>
        </span>
    )
}
