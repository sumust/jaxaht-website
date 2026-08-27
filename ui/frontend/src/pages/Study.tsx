import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
import { Layout } from "../components/Layout"
import { Spinner } from "../components/Spinner"
import { api, ApiError, type PartnerInfo, type StudyStepResponse } from "../api/client"
import { getEnvComponents } from "../envs/registry"
import { GameLog, type LogEntry } from "../envs/hanabi/GameLog"
import type { HanabiState } from "../envs/hanabi/types"

type StudyGameState = {
    study_id: string
    session_id: string
    current_game_index: number
    total_games: number
    num_warmup: number
    is_warmup: boolean
    partner_key: string
    state: Record<string, unknown>
    score: Record<string, unknown>
    done: boolean
    events: LogEntry[]
    lastReward?: number
    session_complete: boolean
    completion_code: string | null
}

export function Study() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [params] = useSearchParams()
    const navigate = useNavigate()
    const prolificPid = params.get("PROLIFIC_PID") ?? ""
    const studyIdParam = params.get("STUDY_ID") ?? ""
    const sessionIdParam = params.get("SESSION_ID") ?? ""

    const [partners, setPartners] = useState<PartnerInfo[] | null>(null)
    const [game, setGame] = useState<StudyGameState | null>(null)
    const [loadingAction, setLoadingAction] = useState(false)
    const [partnerThinking, setPartnerThinking] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [advanceBanner, setAdvanceBanner] = useState<string | null>(null)
    const bootstrappedRef = useRef(false)

    // partner roster for display-name lookup
    useEffect(() => {
        api.partners(env).then((r) => setPartners(r.partners)).catch(() => {})
    }, [env])

    // bootstrap: start a new study session on mount
    useEffect(() => {
        if (bootstrappedRef.current) return
        bootstrappedRef.current = true
        ;(async () => {
            try {
                setLoadingAction(true)
                setError(null)
                const resp = await api.studyStart(env, {
                    prolific_pid: prolificPid || undefined,
                    study_id: studyIdParam || undefined,
                    prolific_session_id: sessionIdParam || undefined,
                    data_source: prolificPid ? "prolific" : "test",
                })
                setGame({
                    study_id: resp.study_id,
                    session_id: resp.session_id,
                    current_game_index: resp.current_game_index,
                    total_games: resp.total_games,
                    num_warmup: resp.num_warmup,
                    is_warmup: resp.is_warmup,
                    partner_key: resp.partner_key,
                    state: resp.state,
                    score: resp.score,
                    done: false,
                    events: [],
                    session_complete: false,
                    completion_code: null,
                })
            } catch (e) {
                setError(e instanceof ApiError ? e.message : String(e))
            } finally {
                setLoadingAction(false)
            }
        })()
    }, [env, prolificPid, studyIdParam, sessionIdParam])

    const sendAction = useCallback(async (action: unknown) => {
        if (!game || game.done || game.session_complete || loadingAction) return
        const studyId = game.study_id
        try {
            setError(null)
            setLoadingAction(true)
            setPartnerThinking(true)
            const resp: StudyStepResponse & {
                reward?: number
                done?: boolean
                partner_acted?: boolean
                events?: LogEntry[]
            } = await api.studyStep(env, studyId, action as Record<string, unknown>) as any
            const newEvents = (resp.events ?? []) as LogEntry[]

            if (resp.session_complete && resp.completion_code) {
                // freeze final state, navigate after a beat
                setGame((prev) => prev ? {
                    ...prev,
                    state: resp.state,
                    score: resp.score,
                    done: true,
                    events: [...prev.events, ...newEvents],
                    session_complete: true,
                    completion_code: resp.completion_code,
                } : null)
                setTimeout(() => {
                    navigate(`/${env}/study/complete?code=${encodeURIComponent(resp.completion_code!)}`)
                }, 2000)
                return
            }

            if (resp.game_just_advanced) {
                // server rotated to next game; reset local events but keep study_id
                setAdvanceBanner(`Game ${(resp.prev_game_index ?? 0) + 1} done. Starting game ${resp.current_game_index + 1}...`)
                setTimeout(() => setAdvanceBanner(null), 2500)
                setGame({
                    study_id: resp.study_id,
                    session_id: resp.session_id,
                    current_game_index: resp.current_game_index,
                    total_games: resp.total_games,
                    num_warmup: resp.num_warmup,
                    is_warmup: resp.is_warmup,
                    partner_key: resp.partner_key,
                    state: resp.state,
                    score: resp.score,
                    done: false,
                    events: [],
                    session_complete: false,
                    completion_code: null,
                })
                return
            }

            // same game continues
            setGame((prev) => prev ? {
                ...prev,
                session_id: resp.session_id,
                state: resp.state,
                score: resp.score,
                done: Boolean(resp.done),
                lastReward: resp.reward,
                events: [...prev.events, ...newEvents],
                partner_key: resp.partner_key,
            } : null)
        } catch (e) {
            if (e instanceof ApiError && e.status === 404) {
                setError("Study session expired. Reload to start a new one.")
                setGame(null)
            } else {
                setError(e instanceof ApiError ? e.message : String(e))
            }
        } finally {
            setLoadingAction(false)
            setPartnerThinking(false)
        }
    }, [env, game, loadingAction, navigate])

    const components = useMemo(() => {
        try { return getEnvComponents(env) } catch { return null }
    }, [env])

    // ESC = abort study (would navigate back; for prolific workers this is the "give up" path)
    useEffect(() => {
        if (!game || game.session_complete) return
        const handler = (ev: KeyboardEvent) => {
            if (ev.target instanceof HTMLInputElement) return
            // No ESC quit in study mode - workers need to finish to get paid.
            // Leaving the listener as a no-op so users feel resistance if they try.
        }
        window.addEventListener("keydown", handler)
        return () => window.removeEventListener("keydown", handler)
    }, [game])

    if (!components) {
        return (
            <Layout title={`Unknown env: ${envDisplayName(env)}`}>
                <p className="text-sm text-ink-400">
                    Frontend adapter not registered for "{env}".
                </p>
            </Layout>
        )
    }

    const { Board } = components
    const partnerDisplayName = partners?.find((p) => p.key === game?.partner_key)?.display_name ?? game?.partner_key ?? "..."
    const gameIdxDisplay = game ? game.current_game_index + 1 : 0
    const phaseLabel = game?.is_warmup ? "warmup" : "real"

    return (
        <Layout
            title={game ? `${envDisplayName(env)} study — Game ${gameIdxDisplay} of ${game.total_games}` : `Loading ${envDisplayName(env)} study...`}
            subtitle={
                game
                    ? (
                        <span>
                            <span className={`pill mr-2 ${game.is_warmup ? "bg-ink-800 text-ink-400" : "bg-hanabi-green/20 text-hanabi-green"}`}>
                                {phaseLabel}
                            </span>
                            vs <span className="text-ink-200 font-medium">{partnerDisplayName}</span>
                            {partnerThinking && <span className="ml-2 text-ink-500">· partner thinking</span>}
                        </span>
                    )
                    : "starting up..."
            }
            right={game && (
                <div className="flex items-center gap-3">
                    {typeof game.score.score === "number" && typeof game.score.max_score === "number" && (
                        <div className="pill bg-ink-800 text-ink-200 text-sm font-mono tabular-nums">
                            {String(game.score.score)} / {String(game.score.max_score)}
                        </div>
                    )}
                </div>
            )}
        >
            {error && (
                <div className="card-surface p-4 text-sm text-hanabi-red mb-6 animate-slide-up">
                    {error}
                </div>
            )}

            {advanceBanner && (
                <div className="card-surface p-4 text-sm text-hanabi-green mb-6 animate-slide-up">
                    {advanceBanner}
                </div>
            )}

            {!game && !error && (
                <div className="card-surface p-8 text-center">
                    <Spinner size="md" label="Booting env - first game takes ~10s while JAX compiles" />
                </div>
            )}

            {game && !game.session_complete && (
                <StudyProgressBar
                    currentIdx={game.current_game_index}
                    totalGames={game.total_games}
                    numWarmup={game.num_warmup}
                />
            )}

            {game && (
                <div className="grid grid-cols-12 gap-6 animate-fade-in">
                    <div className="col-span-12 xl:col-span-9 space-y-4">
                        <Board
                            state={components.parseState(game.state)}
                            onAction={sendAction}
                            disabled={loadingAction || game.done || game.session_complete}
                        />
                        {game.done && !game.session_complete && (
                            <GameAdvance
                                gameIdx={game.current_game_index + 1}
                                totalGames={game.total_games}
                                isWarmup={game.is_warmup}
                                score={game.score}
                            />
                        )}
                        {game.session_complete && game.completion_code && (
                            <StudyDone code={game.completion_code} />
                        )}
                    </div>
                    <aside className="col-span-12 xl:col-span-3">
                        <section className="card-surface p-6 sticky top-20">
                            <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-3">
                                Game log
                            </h3>
                            {env === "hanabi" || env === "mini-hanabi" ? (
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

function GenericLog({ events }: { events: LogEntry[] }) {
    if (events.length === 0) {
        return <p className="text-xs text-ink-500 italic">No events yet.</p>
    }
    const last = events.slice(-10).reverse()
    return (
        <ol className="space-y-1 text-sm">
            {last.map((e, i) => (
                <li key={i} className="text-ink-300 font-mono text-xs">
                    <span className={e.player === 0 ? "text-hanabi-blue" : "text-hanabi-yellow"}>
                        {e.player === 0 ? "you" : "partner"}
                    </span>
                    {" → "}
                    {JSON.stringify(e).slice(0, 80)}
                </li>
            ))}
        </ol>
    )
}

// Per-session progress bar showing where you are in the warmup → real → done sequence.
// Mirrors Johnny's session progress strip in human_data_collecting/templates/index.html.
function StudyProgressBar({
    currentIdx, totalGames, numWarmup,
}: { currentIdx: number; totalGames: number; numWarmup: number }) {
    return (
        <div className="card-surface p-4 mb-4">
            <div className="flex items-center justify-between mb-2 text-xs text-ink-400">
                <span>
                    <span className="text-ink-200 font-medium">Game {currentIdx + 1}</span>
                    <span className="text-ink-500"> / {totalGames}</span>
                </span>
                <span className="text-ink-500">
                    {numWarmup} warmup, then {totalGames - numWarmup} counted
                </span>
            </div>
            <div className="flex gap-1">
                {Array.from({ length: totalGames }, (_, i) => {
                    const isWarmup = i < numWarmup
                    const isPast = i < currentIdx
                    const isCurrent = i === currentIdx
                    const bg = isPast
                        ? (isWarmup ? "bg-ink-700" : "bg-hanabi-green/60")
                        : isCurrent
                        ? (isWarmup ? "bg-ink-500 animate-pulse" : "bg-hanabi-green animate-pulse")
                        : (isWarmup ? "bg-ink-800" : "bg-ink-800/40 border border-hanabi-green/20")
                    return (
                        <div
                            key={i}
                            className={`flex-1 h-2 rounded ${bg}`}
                            title={`Game ${i + 1} ${isWarmup ? "(warmup)" : "(counted)"}`}
                        />
                    )
                })}
            </div>
        </div>
    )
}


function GameAdvance({
    gameIdx, totalGames, isWarmup, score,
}: {
    gameIdx: number
    totalGames: number
    isWarmup: boolean
    score: Record<string, unknown>
}) {
    return (
        <div className="card-surface p-8 text-center space-y-4 animate-slide-up">
            <div className="text-sm uppercase tracking-wider text-ink-500">
                {isWarmup ? "warmup done" : `game ${gameIdx} of ${totalGames} done`}
            </div>
            <div className="text-5xl font-bold text-ink-100 tabular-nums">
                {typeof score.score === "number" ? score.score : "..."}
            </div>
            {typeof score.max_score === "number" && (
                <div className="text-sm text-ink-400">out of {score.max_score}</div>
            )}
            <p className="text-sm text-ink-500 pt-2">
                advancing to the next game...
            </p>
        </div>
    )
}

function StudyDone({ code }: { code: string }) {
    return (
        <div className="card-surface p-8 text-center space-y-4 animate-slide-up">
            <div className="text-sm uppercase tracking-wider text-hanabi-green">study complete</div>
            <p className="text-sm text-ink-300">your completion code:</p>
            <code className="block px-4 py-3 rounded-lg bg-ink-900 border border-ink-700 text-xl font-mono text-ink-100 tracking-widest">
                {code}
            </code>
            <p className="text-xs text-ink-500">redirecting to completion page...</p>
        </div>
    )
}
