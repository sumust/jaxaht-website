import { useCallback, useEffect, useMemo, useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { useParams } from "react-router-dom"
import { Layout } from "../components/Layout"
import { EnvTips } from "../components/EnvTips"
import { PartnerPicker } from "../components/PartnerPicker"
import { Spinner } from "../components/Spinner"
import { api, ApiError, type PartnerInfo } from "../api/client"
import { getEnvComponents } from "../envs/registry"
import { GameLog, type LogEntry } from "../envs/hanabi/GameLog"
import type { HanabiState } from "../envs/hanabi/types"

type GameState = {
    sessionId: string
    state: Record<string, unknown>
    score: Record<string, unknown>
    done: boolean
    events: LogEntry[]
    lastReward?: number
    partnerKey: string
}

export function Play() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [partners, setPartners] = useState<PartnerInfo[] | null>(null)
    const [selectedPartner, setSelectedPartner] = useState<string>("")
    const [game, setGame] = useState<GameState | null>(null)
    const [loadingAction, setLoadingAction] = useState(false)
    const [partnerThinking, setPartnerThinking] = useState(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        api.partners(env)
            .then((r) => {
                setPartners(r.partners)
                if (r.partners.length > 0) {
                    setSelectedPartner(r.partners[0].key)
                }
            })
            .catch((e) => setError(e.message ?? String(e)))
    }, [env])

    const startGame = async () => {
        try {
            setError(null)
            setLoadingAction(true)
            const resp = await api.newGame(env, selectedPartner)
            setGame({
                sessionId: resp.session_id,
                state: resp.state,
                score: resp.score,
                done: false,
                events: [],
                partnerKey: selectedPartner,
            })
        } catch (e) {
            setError(e instanceof ApiError ? e.message : String(e))
        } finally {
            setLoadingAction(false)
        }
    }

    const sendAction = useCallback(async (action: unknown) => {
        // Read current game from a ref-like dance via functional state
        // to avoid stale closures across rapid clicks.
        if (!game || game.done || loadingAction) return
        const sessionId = game.sessionId
        try {
            setError(null)
            setLoadingAction(true)
            setPartnerThinking(true)
            const resp = await api.step(env, sessionId, action as Record<string, unknown>)
            const newEvents = (resp.events ?? []) as LogEntry[]
            setGame((prev) => {
                if (!prev || prev.sessionId !== sessionId) return prev
                return {
                    ...prev,
                    state: resp.state,
                    score: resp.score,
                    done: resp.done,
                    lastReward: resp.reward,
                    events: [...prev.events, ...newEvents],
                }
            })
        } catch (e) {
            if (e instanceof ApiError && e.status === 404) {
                setGame(null)
                setError("Session expired. Start a new game.")
            } else {
                setError(e instanceof ApiError ? e.message : String(e))
            }
        } finally {
            setLoadingAction(false)
            setPartnerThinking(false)
        }
    }, [env, game, loadingAction])

    const quit = () => { setGame(null) }

    const components = useMemo(() => {
        try { return getEnvComponents(env) } catch { return null }
    }, [env])

    // Keyboard shortcuts - only active during play (env-specific map
    // would live in the env registry; for now Hanabi-leaning defaults).
    useEffect(() => {
        if (!game || game.done) return
        const handler = (ev: KeyboardEvent) => {
            if (ev.target instanceof HTMLInputElement) return
            if (ev.key === "Escape") quit()
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [game])

    if (!components) {
        return (
            <Layout title={`Unknown env: ${envDisplayName(env)}`}>
                <p className="text-sm text-ink-400">
                    Frontend adapter not registered. Add one in <code className="text-ink-200 font-mono text-xs">frontend/src/envs/registry.ts</code>.
                </p>
            </Layout>
        )
    }

    const { Board } = components

    return (
        <Layout
            title={game ? `Playing ${envDisplayName(env)}` : `Set up a ${envDisplayName(env)} game`}
            subtitle={
                game
                    ? (
                        <span>
                            vs <span className="text-ink-200 font-medium">
                                {partners?.find((p) => p.key === game.partnerKey)?.display_name ?? game.partnerKey}
                            </span>
                            {partnerThinking && <span className="ml-2 text-ink-500">· partner thinking</span>}
                        </span>
                    )
                    : "Choose a partner to play against"
            }
            right={game && (
                <div className="flex items-center gap-3">
                    {typeof game.score.score === "number" && typeof game.score.max_score === "number" && (
                        <div className="pill bg-ink-800 text-ink-200 text-sm font-mono tabular-nums">
                            {String(game.score.score)} / {String(game.score.max_score)}
                        </div>
                    )}
                    <button className="btn-ghost" onClick={quit}>End</button>
                </div>
            )}
        >
            {error && (
                <div className="card-surface p-4 text-sm text-hanabi-red mb-6 animate-slide-up">
                    {error}
                </div>
            )}

            {!game && partners && (
                <div className="space-y-6">
                    <EnvTips env={env} />
                    <PartnerPicker
                        partners={partners}
                        selected={selectedPartner}
                        onSelect={setSelectedPartner}
                    />
                    <div className="flex justify-end items-center gap-3">
                        {loadingAction && (
                            <Spinner size="sm" label="Booting env - first game takes ~10s while JAX compiles" />
                        )}
                        <button
                            className="btn-primary"
                            onClick={startGame}
                            disabled={!selectedPartner || loadingAction}
                        >
                            {loadingAction ? "Starting…" : "Start game"}
                        </button>
                    </div>
                </div>
            )}

            {!game && !partners && !error && (
                <div className="text-sm text-ink-500">Loading partners…</div>
            )}

            {game && (
                <div className="grid grid-cols-12 gap-6 animate-fade-in">
                    <div className="col-span-12 xl:col-span-9 space-y-4">
                        <Board
                            state={components.parseState(game.state)}
                            onAction={sendAction}
                            disabled={loadingAction || game.done}
                        />
                        {game.done && (
                            <GameOver
                                env={env}
                                sessionId={game.sessionId}
                                score={game.score}
                                onRestart={() => { setGame(null) }}
                            />
                        )}
                    </div>
                    <aside className="col-span-12 xl:col-span-3">
                        <section className="card-surface p-6 sticky top-20">
                            <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-3">
                                Game log
                            </h3>
                            {(env === "hanabi" || env === "mini-hanabi") ? (
                                <GameLog
                                    entries={game.events}
                                    state={components.parseState(game.state) as HanabiState}
                                />
                            ) : (
                                <GenericLog events={game.events} />
                            )}
                        </section>
                    </aside>
                </div>
            )}
        </Layout>
    )
}

// Narrate a single event in plain English. Fallback used by LBF + Overcooked.
function describeEvent(e: any): string {
    const k = String(e.kind ?? "")
    if (k === "move") {
        const dir = e.direction ?? ""
        if (dir === "stay" || dir === "wait" || dir === "noop") return "stayed"
        if (dir === "interact") return "interacted"
        return `moved ${dir}`
    }
    if (k === "interact") return "interacted"
    if (k === "load") return `loaded fruit ${e.fruit_idx ?? ""}`.trim()
    if (k === "delivery") return "delivered a soup"
    if (k === "pickup") return `picked up ${e.what ?? "item"}`
    if (k === "drop") return `dropped ${e.what ?? "item"}`
    if (k === "noop") return "did nothing"
    return k || "acted"
}

function GenericLog({ events }: { events: LogEntry[] }) {
    if (events.length === 0) {
        return <p className="text-xs text-ink-500 italic">No events yet.</p>
    }
    const last = events.slice(-10).reverse()
    return (
        <ol className="space-y-1.5 text-sm">
            {last.map((e, i) => (
                <li key={i} className="text-ink-300 text-sm">
                    <span className={`font-medium ${e.player === 0 ? "text-hanabi-blue" : "text-hanabi-yellow"}`}>
                        {e.player === 0 ? "You" : "Partner"}
                    </span>
                    {" "}
                    <span className="text-ink-400">{describeEvent(e)}</span>
                    {(e as any).reward && (e as any).reward !== 0 && (
                        <span className="ml-1.5 text-hanabi-green text-xs">+{(e as any).reward}</span>
                    )}
                </li>
            ))}
        </ol>
    )
}


function GameOver({
    env, sessionId, score, onRestart,
}: {
    env: string
    sessionId: string
    score: Record<string, unknown>
    onRestart: () => void
}) {
    const [agentName, setAgentName] = useState("Anonymous")
    const [saved, setSaved] = useState<string | null>(null)
    const [saving, setSaving] = useState(false)
    const [err, setErr] = useState<string | null>(null)

    const save = async () => {
        try {
            setSaving(true)
            setErr(null)
            const resp = await api.saveTrajectory(env, sessionId, agentName)
            setSaved(resp.trajectory_id)
        } catch (e) {
            setErr(e instanceof ApiError ? e.message : String(e))
        } finally {
            setSaving(false)
        }
    }

    return (
        <div className="card-surface p-8 text-center space-y-4 animate-slide-up">
            <div className="text-sm uppercase tracking-wider text-ink-500">Game over</div>
            <div className="text-5xl font-bold text-ink-100 tabular-nums">
                {typeof score.score === "number" ? score.score : "..."}
            </div>
            {typeof score.max_score === "number" && (
                <div className="text-sm text-ink-400">out of {score.max_score}</div>
            )}

            {!saved ? (
                <div className="max-w-sm mx-auto space-y-2 pt-4">
                    <label className="text-xs uppercase tracking-wider text-ink-500 block text-left">
                        Save this game under:
                    </label>
                    <input
                        className="w-full bg-ink-900 border border-ink-700 rounded-lg px-3 py-2 text-ink-100 focus:outline-none focus:border-ink-500"
                        value={agentName}
                        onChange={(e) => setAgentName(e.target.value)}
                        maxLength={64}
                    />
                    {err && <p className="text-xs text-hanabi-red">{err}</p>}
                    <div className="flex gap-2 justify-center pt-2">
                        <button className="btn-primary" onClick={save} disabled={saving}>
                            {saving ? "Saving…" : "Save trajectory"}
                        </button>
                        <button className="btn-outline" onClick={onRestart}>
                            Play again
                        </button>
                    </div>
                </div>
            ) : (
                <div className="space-y-2 pt-4">
                    <div className="text-sm text-hanabi-green">
                        Saved as <code className="font-mono text-ink-200">{saved}</code>
                    </div>
                    <button className="btn-primary mt-2" onClick={onRestart}>
                        Play again
                    </button>
                </div>
            )}
        </div>
    )
}
