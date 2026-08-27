import { useEffect } from "react"
import type { OvercookedAction, OvercookedAgent, OvercookedPot, OvercookedState } from "./types"

type MoveDir = Extract<OvercookedAction, { type: "move" }>["dir"]

// Overcooked board. Walls are dark slate, empty floor is light, pots
// have a stove icon, agents have direction-pointer color blocks with
// inventory badges showing what they're holding (onion / plate / soup).
//
// Keyboard: WASD + space (interact) + q (stay).

type Props = {
    state: OvercookedState
    onAction: (action: OvercookedAction) => void
    disabled?: boolean
}

const AGENT_COLORS = ["bg-blue-500", "bg-amber-500"]
const AGENT_BORDERS = ["border-blue-700", "border-amber-700"]

const HOLDING_LABELS: Record<number, string> = {
    0: "",
    1: "🧅",
    2: "🍽",
    3: "🍲",
}

const DIR_GLYPHS: Record<number, string> = {
    0: "↑",
    1: "↓",
    2: "→",
    3: "←",
}

export function OvercookedBoard({ state, onAction, disabled }: Props) {
    useEffect(() => {
        if (disabled) return
        const handler = (ev: KeyboardEvent) => {
            if (ev.target instanceof HTMLInputElement) return
            const key = ev.key.toLowerCase()
            const dirMap: Record<string, MoveDir> = {
                w: "up", arrowup: "up",
                s: "down", arrowdown: "down",
                a: "left", arrowleft: "left",
                d: "right", arrowright: "right",
                " ": "interact",
                q: "stay",
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

    const cellPx = Math.max(40, Math.min(80, 480 / Math.max(state.width, state.height)))
    const boardW = cellPx * state.width
    const boardH = cellPx * state.height

    const wallAt = (x: number, y: number) => {
        if (state.wall_map.length === 0) return false
        return Boolean(state.wall_map[y]?.[x])
    }
    const potAt = new Map<string, OvercookedPot>()
    for (const p of state.pots) potAt.set(`${p.x},${p.y}`, p)

    const agentAt = new Map<string, { agent: OvercookedAgent; idx: number }>()
    state.agents.forEach((a, idx) => agentAt.set(`${a.x},${a.y}`, { agent: a, idx }))

    // Compute the "facing tile" for each agent — the tile they'd interact with on space.
    // dir: 0=up, 1=down, 2=right, 3=left
    const DIR_OFFSET: Record<number, { dx: number; dy: number }> = {
        0: { dx: 0, dy: -1 }, 1: { dx: 0, dy: 1 }, 2: { dx: 1, dy: 0 }, 3: { dx: -1, dy: 0 },
    }
    const facingHints = new Map<string, number>()  // "x,y" -> agent_idx (highlight color)
    state.agents.forEach((a, idx) => {
        const off = DIR_OFFSET[a.dir] ?? { dx: 0, dy: 0 }
        facingHints.set(`${a.x + off.dx},${a.y + off.dy}`, idx)
    })

    return (
        <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-9 space-y-4">
                <section className="card-surface p-8 flex items-center justify-center">
                    <div
                        className="relative border border-ink-800 rounded-lg overflow-hidden bg-ink-900"
                        style={{ width: boardW, height: boardH }}
                    >
                        {Array.from({ length: state.height }, (_, y) => (
                            Array.from({ length: state.width }, (_, x) => {
                                const wall = wallAt(x, y)
                                const pot = potAt.get(`${x},${y}`)
                                const agent = agentAt.get(`${x},${y}`)
                                const facingHint = facingHints.get(`${x},${y}`)  // which agent is looking at this tile
                                const facingRing = facingHint === 0
                                    ? "inset 0 0 0 3px rgba(59,130,246,0.55)"     // blue (player 0)
                                    : facingHint === 1
                                    ? "inset 0 0 0 3px rgba(245,158,11,0.55)"     // amber (player 1)
                                    : undefined
                                return (
                                    <div
                                        key={`${x}-${y}`}
                                        className={`absolute flex items-center justify-center text-xs ${wall ? "bg-slate-700" : "bg-stone-200"}`}
                                        style={{
                                            left: x * cellPx,
                                            top: y * cellPx,
                                            width: cellPx,
                                            height: cellPx,
                                            border: "1px solid rgba(0,0,0,0.08)",
                                            boxShadow: facingRing,
                                        }}
                                    >
                                        {pot && (
                                            <div className="text-2xl" title="pot">🍳</div>
                                        )}
                                        {agent && (
                                            <div
                                                className={`relative rounded-full ${AGENT_COLORS[agent.idx % 2]} border-2 ${AGENT_BORDERS[agent.idx % 2]} flex items-center justify-center text-white font-bold`}
                                                style={{
                                                    width: cellPx * 0.7,
                                                    height: cellPx * 0.7,
                                                }}
                                            >
                                                <span className="text-base">{DIR_GLYPHS[agent.agent.dir] ?? "?"}</span>
                                                {agent.agent.holding > 0 && (
                                                    <span
                                                        className="absolute -top-1 -right-1 text-base bg-white rounded-full"
                                                        style={{ width: cellPx * 0.32, height: cellPx * 0.32, lineHeight: `${cellPx * 0.32}px`, textAlign: "center" }}
                                                        title={`holding ${HOLDING_LABELS[agent.agent.holding] || `inv${agent.agent.holding}`}`}
                                                    >
                                                        {HOLDING_LABELS[agent.agent.holding] ?? "?"}
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )
                            })
                        ))}
                    </div>
                </section>

                <section className="card-surface p-4 space-y-2">
                    <h3 className="font-semibold mb-1">controls</h3>
                    <p className="text-sm text-ink-400">
                        WASD or arrow keys to move. Space to interact. Q to stay.
                    </p>
                    <p className="text-xs text-ink-500 leading-relaxed">
                        <span className="text-ink-300 font-medium">Heads up:</span> Overcooked uses turn-then-move semantics.
                        First press in a new direction <em>rotates</em> you to face that way (look at the arrow on your character);
                        the second press <em>moves</em> you forward. Space only works when you're directly facing
                        a pot / onion stack / plate stack / delivery counter — if it seems to "do nothing", you
                        probably aren't facing the right tile.
                    </p>
                    <p className="text-xs text-ink-500">
                        Recipe: pick up onion → drop in pot → wait for soup → grab plate → plate soup → deliver.
                    </p>
                </section>
            </div>

            <aside className="col-span-12 lg:col-span-3 space-y-4">
                <section className="card-surface p-4">
                    <h3 className="font-semibold mb-2">layout</h3>
                    <p className="text-sm text-ink-300 font-mono">{state.layout}</p>
                    <p className="text-xs text-ink-500 mt-1">
                        {state.width} × {state.height} grid, step {state.step_count}/400
                    </p>
                </section>
                <section className="card-surface p-4">
                    <h3 className="font-semibold mb-2">legend</h3>
                    <ul className="text-sm text-ink-300 space-y-1">
                        <li>🍳 pot — drop onions here</li>
                        <li>🧅 onion (in inventory)</li>
                        <li>🍽 plate (in inventory)</li>
                        <li>🍲 soup (in inventory)</li>
                    </ul>
                </section>
            </aside>
        </div>
    )
}
