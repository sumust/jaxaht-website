// Right-panel game log. Narrates what just happened each turn in
// plain English so the human doesn't have to keep a mental tape.
// Hanabi is hard enough without reverse-engineering the last three
// turns from state diffs.

import { COLOR_TEXT, rankGlyph } from "./cards"
import type { HanabiState } from "./types"

export type LogEntry = {
    player: 0 | 1
    kind: "play" | "discard" | "hint_color" | "hint_rank" | "noop"
    slot?: number
    color?: number
    rank?: number
    bombed?: boolean
    scored?: boolean
    revealed?: { color?: number; rank?: number }
}

type Props = {
    entries: LogEntry[]
    state: HanabiState
    // optional override for player labels. Defaults to ["You", "Partner"] for /play mode.
    // /demo mode passes the actual policy names.
    playerNames?: [string, string]
}

export function GameLog({ entries, state, playerNames }: Props) {
    if (entries.length === 0) {
        return (
            <div className="text-xs text-ink-500 italic">
                Game log appears here as you play.
            </div>
        )
    }
    // Most recent first, capped to last 12. React key derived from
    // absolute position in the full history so it stays stable across
    // renders even as new entries prepend the visible slice.
    const total = entries.length
    const items = entries.slice(-12)
    return (
        <ol className="space-y-2">
            {items.slice().reverse().map((entry, revIdx) => {
                const absoluteIdx = total - 1 - revIdx
                return (
                    <li key={absoluteIdx} className="text-sm animate-fade-in">
                        <LogLine entry={entry} state={state} playerNames={playerNames} />
                    </li>
                )
            })}
        </ol>
    )
}

function LogLine({ entry, state, playerNames }: { entry: LogEntry; state: HanabiState; playerNames?: [string, string] }) {
    const [name0, name1] = playerNames ?? ["You", "Partner"]
    const who = entry.player === 0 ? name0 : name1
    const whoClass = entry.player === 0 ? "text-hanabi-blue" : "text-hanabi-yellow"
    const base = <span className={`font-medium ${whoClass}`}>{who}</span>

    if (entry.kind === "play") {
        const color = entry.color ?? entry.revealed?.color
        const rank = entry.rank ?? entry.revealed?.rank
        const card = color !== undefined && rank !== undefined
            ? <span className={`font-mono ${COLOR_TEXT[color]}`}>
                  {state.color_names[color]} {rankGlyph(rank)}
              </span>
            : <span>slot {(entry.slot ?? 0) + 1}</span>
        if (entry.bombed) {
            return <div>{base} played {card}, <span className="text-hanabi-red">bombed</span></div>
        }
        if (entry.scored) {
            return <div>{base} played {card}, <span className="text-hanabi-green">scored</span></div>
        }
        return <div>{base} played {card}</div>
    }

    if (entry.kind === "discard") {
        return (
            <div>
                {base} discarded slot {(entry.slot ?? 0) + 1}
            </div>
        )
    }

    if (entry.kind === "hint_color" && entry.color !== undefined) {
        return (
            <div>
                {base} hinted <span className={COLOR_TEXT[entry.color]}>{state.color_names[entry.color]}</span>
            </div>
        )
    }

    if (entry.kind === "hint_rank" && entry.rank !== undefined) {
        return (
            <div>
                {base} hinted <span className="font-mono">{rankGlyph(entry.rank)}</span>
            </div>
        )
    }

    return <div className="text-ink-500">{base} passed</div>
}
