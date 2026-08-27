import { useEffect, useRef, useState } from "react"
import type { HanabiAction, HanabiState } from "./types"
import { CardSlot } from "./CardSlot"
import { FireworksTracker } from "./FireworksTracker"
import { TokenRow } from "./Tokens"
import { DiscardPile } from "./DiscardPile"
import { HintMenu } from "./HintMenu"

type Props = {
    state: HanabiState
    onAction: (action: HanabiAction) => void
    disabled?: boolean
}

// Main Hanabi play view. Four functional regions:
//   - top-left: partner hand (visible to human)
//   - top-right: fireworks + tokens
//   - bottom:   human hand (beliefs only)
//   - right panel: discard pile + deck count
//
// Interaction:
//   click human slot -> prompts play/discard via ControlBar
//   select partner slot + "hint" -> opens HintMenu to pick color/rank
//
// Human controls (ControlBar.tsx) live separately so the Board stays
// focused on rendering state.

export function HanabiBoard({ state, onAction, disabled }: Props) {
    const [selectedHumanSlot, setSelectedHumanSlot] = useState<number | null>(null)
    const [selectedPartnerSlot, setSelectedPartnerSlot] = useState<number | null>(null)
    const [showHintMenu, setShowHintMenu] = useState(false)

    const isHumanTurn = state.current_player === 0 && !disabled

    const handleHumanSlotClick = (slot: number) => {
        if (!isHumanTurn) return
        setSelectedHumanSlot(slot)
        setSelectedPartnerSlot(null)
        setShowHintMenu(false)
    }

    const handlePartnerSlotClick = (slot: number) => {
        if (!isHumanTurn) return
        setSelectedPartnerSlot(slot)
        setSelectedHumanSlot(null)
        setShowHintMenu(true)
    }

    const doPlay = () => {
        if (selectedHumanSlot === null) return
        onAction({ type: "play", slot: selectedHumanSlot })
        setSelectedHumanSlot(null)
    }
    const doDiscard = () => {
        if (selectedHumanSlot === null) return
        onAction({ type: "discard", slot: selectedHumanSlot })
        setSelectedHumanSlot(null)
    }
    const doHint = (a: HanabiAction) => {
        onAction(a)
        setSelectedPartnerSlot(null)
        setShowHintMenu(false)
    }

    // Keyboard shortcuts.
    //   1-5       select human hand slot
    //   p / Enter play selected slot
    //   d / Backspace  discard selected slot
    //   Escape    clear selection
    //
    // We stash the per-frame closures into a ref so the keydown
    // handler always sees the latest state, selection, and onAction
    // without needing to re-register the listener on every render.
    const latest = useRef({
        state, selectedHumanSlot, onAction, isHumanTurn,
    })
    latest.current = { state, selectedHumanSlot, onAction, isHumanTurn }

    useEffect(() => {
        const handler = (ev: KeyboardEvent) => {
            const { state, selectedHumanSlot, onAction, isHumanTurn } = latest.current
            if (!isHumanTurn) return
            if (ev.target instanceof HTMLInputElement) return
            if (ev.metaKey || ev.ctrlKey || ev.altKey) return

            const key = ev.key
            const slotIdx = "12345".indexOf(key)
            if (slotIdx >= 0 && slotIdx < state.hand_size) {
                ev.preventDefault()
                setSelectedHumanSlot(slotIdx)
                setSelectedPartnerSlot(null)
                setShowHintMenu(false)
                return
            }
            if ((key === "p" || key === "P" || key === "Enter") && selectedHumanSlot !== null) {
                ev.preventDefault()
                onAction({ type: "play", slot: selectedHumanSlot })
                setSelectedHumanSlot(null)
                return
            }
            if ((key === "d" || key === "D" || key === "Backspace") && selectedHumanSlot !== null) {
                ev.preventDefault()
                onAction({ type: "discard", slot: selectedHumanSlot })
                setSelectedHumanSlot(null)
                return
            }
            if (key === "Escape") {
                ev.preventDefault()
                setSelectedHumanSlot(null)
                setSelectedPartnerSlot(null)
                setShowHintMenu(false)
            }
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [])  // eslint-disable-line react-hooks/exhaustive-deps

    return (
        <div className="grid grid-cols-12 gap-6">
            {/* === Partner hand + context === */}
            <div className="col-span-12 lg:col-span-8 space-y-4">
                <section className="card-surface p-6 space-y-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                                Partner hand
                            </h3>
                            <p className="text-xs text-ink-500 mt-1">
                                Click a card to give a hint
                            </p>
                        </div>
                        <TurnIndicator current={state.current_player} />
                    </div>
                    <div className="flex gap-2 flex-wrap">
                        {state.partner_hand.map((card, i) => (
                            <CardSlot
                                key={i}
                                card={card}
                                slotIndex={i}
                                selected={selectedPartnerSlot === i}
                                onClick={() => handlePartnerSlotClick(i)}
                                disabled={!isHumanTurn || state.info_tokens === 0}
                            />
                        ))}
                    </div>
                    {state.info_tokens === 0 && (
                        <p className="text-xs text-hanabi-red">
                            No info tokens; you must play or discard instead of hinting.
                        </p>
                    )}
                </section>

                {/* === Your hand (beliefs only) === */}
                <section className="card-surface p-6 space-y-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                                Your hand
                            </h3>
                            <p className="text-xs text-ink-500 mt-1">
                                Click a slot to play or discard it
                            </p>
                        </div>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                        {state.human_beliefs.map((belief, i) => (
                            <CardSlot
                                key={i}
                                belief={belief}
                                slotIndex={i}
                                isHumanHand
                                selected={selectedHumanSlot === i}
                                onClick={() => handleHumanSlotClick(i)}
                                disabled={!isHumanTurn}
                            />
                        ))}
                    </div>
                    {selectedHumanSlot !== null && (
                        <div className="flex gap-2 pt-2 animate-slide-up">
                            <button className="btn-primary" onClick={doPlay} disabled={!isHumanTurn}>
                                Play slot {selectedHumanSlot + 1}
                            </button>
                            <button
                                className="btn-outline"
                                onClick={doDiscard}
                                disabled={!isHumanTurn || state.info_tokens === state.max_info_tokens}
                            >
                                Discard slot {selectedHumanSlot + 1}
                            </button>
                            <button
                                className="btn-ghost"
                                onClick={() => setSelectedHumanSlot(null)}
                            >
                                Cancel
                            </button>
                        </div>
                    )}
                </section>
            </div>

            {/* === Side panel: fireworks / tokens / discard === */}
            <aside className="col-span-12 lg:col-span-4 space-y-4">
                <section className="card-surface p-6 space-y-4">
                    <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                        Fireworks
                    </h3>
                    <FireworksTracker
                        fireworks={state.fireworks}
                        numRanks={state.num_ranks}
                        colorNames={state.color_names}
                    />
                </section>

                <section className="card-surface p-6 space-y-3">
                    <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                        Tokens
                    </h3>
                    <TokenRow label="Info" current={state.info_tokens} max={state.max_info_tokens} variant="info" />
                    <TokenRow label="Life" current={state.life_tokens} max={state.max_life_tokens} variant="life" />
                    <div className="pt-2 flex items-center gap-2 text-xs text-ink-400 font-mono">
                        <span className="uppercase tracking-wider">Deck</span>
                        <span className="tabular-nums text-ink-200">{state.deck_size}</span>
                        <span className="uppercase tracking-wider ml-3">Discarded</span>
                        <span className="tabular-nums text-ink-200">{state.num_cards_discarded}</span>
                    </div>
                </section>

                <section className="card-surface p-6 space-y-3">
                    <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                        Discard pile
                    </h3>
                    <DiscardPile
                        discards={state.discard_pile}
                        numColors={state.num_colors}
                        numRanks={state.num_ranks}
                    />
                </section>
            </aside>

            {/* === Hint menu overlay === */}
            {showHintMenu && selectedPartnerSlot !== null && (
                <HintMenu
                    state={state}
                    partnerSlot={selectedPartnerSlot}
                    onHint={doHint}
                    onCancel={() => {
                        setSelectedPartnerSlot(null)
                        setShowHintMenu(false)
                    }}
                />
            )}

            {/* Keyboard hint strip. Small but present. */}
            <div className="col-span-12 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-500">
                <Kbd keys={["1", "2", "3", "4", "5"]} label="select slot" />
                <Kbd keys={["P"]} label="play" />
                <Kbd keys={["D"]} label="discard" />
                <Kbd keys={["Esc"]} label="cancel" />
            </div>
        </div>
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

function TurnIndicator({ current }: { current: number }) {
    const human = current === 0
    return (
        <div className={`pill ${human ? "bg-hanabi-blue/20 text-hanabi-blue" : "bg-ink-800 text-ink-300"}`}>
            {human ? "Your turn" : "Partner thinking…"}
        </div>
    )
}
