// Per-env "first-time" tip banner shown above the partner picker / board.
// Dismissable per env via localStorage. Keep tips short — full instructions
// belong in the env's Board controls section.

import { useState, useEffect } from "react"

type Tip = {
    title: string
    body: string
}

const TIPS: Record<string, Tip> = {
    "hanabi": {
        title: "First time playing Hanabi here?",
        body: "Cooperative card game with hidden information. You see your partner's hand but not your own. Pick: PLAY a slot, DISCARD a slot (gain info token), or HINT a color/rank (costs an info token). Bombing 3 cards ends the game. Goal: max 25 across 5 fireworks.",
    },
    "mini-hanabi": {
        title: "Mini-Hanabi",
        body: "Same as Hanabi but reduced to 3 colors × 3 ranks. Max score 15. Faster games + cheaper to evaluate against — good for prototyping ego policies before the full game.",
    },
    "lbf": {
        title: "First time playing LBF?",
        body: "Grid foraging. Walk to a fruit and press LOAD when adjacent. Fruits with higher level than your own require both agents to load together. Goal: clear all fruits in 50 steps.",
    },
    "overcooked-v1": {
        title: "First time playing Overcooked?",
        body: "Cooperative cooking on a small grid. Walking semantics: first press in a new direction TURNS you to face that way; second press MOVES you forward. Space INTERACTS with the tile you're facing (pot / onion stack / plate / delivery). Recipe: 3 onions in a pot → wait → plate the soup → deliver.",
    },
}

export function EnvTips({ env }: { env: string }) {
    const tip = TIPS[env]
    const storageKey = `jaxaht.tip.dismissed.${env}`
    const [dismissed, setDismissed] = useState(true)

    useEffect(() => {
        setDismissed(localStorage.getItem(storageKey) === "1")
    }, [storageKey])

    if (!tip || dismissed) return null
    return (
        <section className="card-surface p-4 mb-4 border-l-2 border-ut-orange bg-ut-orange/5">
            <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                    <div className="text-sm font-medium text-ink-100">{tip.title}</div>
                    <p className="text-xs text-ink-400 leading-relaxed">{tip.body}</p>
                </div>
                <button
                    onClick={() => {
                        localStorage.setItem(storageKey, "1")
                        setDismissed(true)
                    }}
                    className="text-xs text-ink-500 hover:text-ink-300"
                    aria-label="dismiss"
                >
                    ✕
                </button>
            </div>
        </section>
    )
}
