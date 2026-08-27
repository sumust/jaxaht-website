import { useEffect, useMemo, useRef, useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { Link, useParams } from "react-router-dom"
import { Layout } from "../components/Layout"
import { Spinner } from "../components/Spinner"
import {
    api,
    ApiError,
    type BuiltinEgo,
    type DemoFrame,
    type PartnerInfo,
} from "../api/client"
import { getEnvComponents } from "../envs/registry"
import { GameLog, type LogEntry } from "../envs/hanabi/GameLog"

type AgentOption = {
    id: string
    display_name: string
    kind: "partner" | "ego" | "upload"
    tag: string
}

export function Demo() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [agents, setAgents] = useState<AgentOption[] | null>(null)
    const [agentA, setAgentA] = useState<string>("")
    const [agentB, setAgentB] = useState<string>("")
    const [seed, setSeed] = useState<number>(42)
    const [maxSteps, setMaxSteps] = useState<number>(100)
    const [frames, setFrames] = useState<DemoFrame[] | null>(null)
    const [frameIdx, setFrameIdx] = useState<number>(0)
    const [running, setRunning] = useState(false)
    const [playing, setPlaying] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const playTimer = useRef<number | null>(null)

    useEffect(() => {
        let cancelled = false
        Promise.allSettled([api.partners(env), api.egos(env), api.uploadedPartners(env)]).then(([pRes, eRes, uRes]) => {
            if (cancelled) return
            const opts: AgentOption[] = []
            if (pRes.status === "fulfilled") {
                for (const p of pRes.value.partners as PartnerInfo[]) {
                    opts.push({
                        id: `partner:${p.key}`,
                        display_name: p.display_name,
                        kind: "partner",
                        tag: p.difficulty,
                    })
                }
            }
            if (eRes.status === "fulfilled") {
                for (const e of eRes.value.egos as BuiltinEgo[]) {
                    opts.push({
                        id: `ego:${e.key}`,
                        display_name: e.display_name,
                        kind: "ego",
                        tag: "builtin ego",
                    })
                }
            }
            if (uRes.status === "fulfilled") {
                for (const u of uRes.value.uploaded) {
                    opts.push({
                        id: `upload:${u.checkpoint_id}`,
                        display_name: u.agent_name,
                        kind: "upload",
                        tag: u.actor_type ?? "uploaded",
                    })
                }
            }
            setAgents(opts)
            if (opts.length > 0) {
                setAgentA(opts[0].id)
                setAgentB(opts[Math.min(1, opts.length - 1)].id)
            }
            const firstErr =
                pRes.status === "rejected" ? pRes.reason :
                eRes.status === "rejected" ? eRes.reason : null
            if (firstErr) {
                setError(firstErr instanceof ApiError ? firstErr.message : String(firstErr))
            }
        })
        return () => { cancelled = true }
    }, [env])

    const components = useMemo(() => {
        try {
            return getEnvComponents(env)
        } catch {
            return null
        }
    }, [env])

    const runDemo = async () => {
        if (!agentA || !agentB) return
        setError(null)
        setRunning(true)
        setPlaying(false)
        if (playTimer.current !== null) {
            window.clearInterval(playTimer.current)
            playTimer.current = null
        }
        try {
            const resp = await api.demo(env, {
                agent_a_id: agentA,
                agent_b_id: agentB,
                seed,
                max_steps: maxSteps,
            })
            setFrames(resp.frames)
            setFrameIdx(0)
        } catch (e) {
            setError(e instanceof ApiError ? e.message : String(e))
        } finally {
            setRunning(false)
        }
    }

    const togglePlay = () => {
        if (!frames || frames.length === 0) return
        if (playing) {
            setPlaying(false)
            if (playTimer.current !== null) {
                window.clearInterval(playTimer.current)
                playTimer.current = null
            }
            return
        }
        setPlaying(true)
        playTimer.current = window.setInterval(() => {
            setFrameIdx((i) => {
                if (!frames || i >= frames.length - 1) {
                    if (playTimer.current !== null) {
                        window.clearInterval(playTimer.current)
                        playTimer.current = null
                    }
                    setPlaying(false)
                    return i
                }
                return i + 1
            })
        }, 600)
    }

    useEffect(() => {
        return () => {
            if (playTimer.current !== null) window.clearInterval(playTimer.current)
        }
    }, [])

    const currentFrame = frames && frames.length > 0 ? frames[frameIdx] : null
    const parsedState = currentFrame && components
        ? components.parseState(currentFrame.state)
        : null
    const agentsById = useMemo(() => {
        const map = new Map<string, AgentOption>()
        for (const a of agents ?? []) map.set(a.id, a)
        return map
    }, [agents])
    const labelA = agentsById.get(agentA)?.display_name ?? agentA
    const labelB = agentsById.get(agentB)?.display_name ?? agentB

    return (
        <Layout
            title={`${envDisplayName(env)} policy demo`}
            subtitle="Pick two policies, run an episode, step through the frames. Held-out partners, built-in egos, and any checkpoints you upload are all selectable."
        >
            <div className="space-y-6">
                <DemoUploadBox env={env} onUploaded={() => {
                    api.uploadedPartners(env).then((r) => {
                        const newOpts: AgentOption[] = (agents ?? []).filter((a) => a.kind !== "upload")
                        for (const u of r.uploaded) {
                            newOpts.push({ id: `upload:${u.checkpoint_id}`, display_name: u.agent_name, kind: "upload", tag: u.actor_type ?? "uploaded" })
                        }
                        setAgents(newOpts)
                        if (r.uploaded.length > 0) setAgentA(`upload:${r.uploaded[r.uploaded.length - 1].checkpoint_id}`)
                    })
                }} />
                <div className="card-surface p-6 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <AgentSelect
                            label="Policy A (player 0)"
                            options={agents}
                            value={agentA}
                            onChange={setAgentA}
                        />
                        <AgentSelect
                            label="Policy B (player 1)"
                            options={agents}
                            value={agentB}
                            onChange={setAgentB}
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <label className="block">
                            <span className="text-xs uppercase tracking-wider text-ink-400">seed</span>
                            <input
                                type="number"
                                value={seed}
                                onChange={(e) => setSeed(parseInt(e.target.value || "0", 10))}
                                className="mt-1 w-full bg-ink-900 border border-ink-700 rounded px-3 py-2 text-ink-100 font-mono text-sm"
                            />
                        </label>
                        <label className="block">
                            <span className="text-xs uppercase tracking-wider text-ink-400">max steps</span>
                            <input
                                type="number"
                                value={maxSteps}
                                onChange={(e) => setMaxSteps(parseInt(e.target.value || "0", 10))}
                                className="mt-1 w-full bg-ink-900 border border-ink-700 rounded px-3 py-2 text-ink-100 font-mono text-sm"
                            />
                        </label>
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={runDemo}
                            disabled={running || !agentA || !agentB}
                            className="btn-primary"
                        >
                            {running ? "Running..." : "Run demo"}
                        </button>
                        {frames && (
                            <span className="text-xs text-ink-400">
                                {labelA} <span className="text-ink-500">vs</span> {labelB} — {frames.length} frames
                            </span>
                        )}
                    </div>
                    {error && <div className="text-sm text-red-400 font-mono">{error}</div>}
                </div>

                {running && (
                    <div className="card-surface p-8 flex items-center justify-center">
                        <Spinner />
                    </div>
                )}

                {!running && frames && currentFrame && components && (
                    <div className="card-surface p-6 space-y-4">
                        <div className="flex items-center justify-between flex-wrap gap-3">
                            <div className="flex items-center gap-3">
                                <button onClick={togglePlay} className="btn-outline">
                                    {playing ? "Pause" : "Play"}
                                </button>
                                <button
                                    onClick={() => setFrameIdx((i) => Math.max(0, i - 1))}
                                    disabled={frameIdx === 0}
                                    className="btn-outline"
                                >
                                    Prev
                                </button>
                                <button
                                    onClick={() => setFrameIdx((i) => Math.min(frames.length - 1, i + 1))}
                                    disabled={frameIdx >= frames.length - 1}
                                    className="btn-outline"
                                >
                                    Next
                                </button>
                            </div>
                            <div className="text-xs font-mono text-ink-400">
                                frame {frameIdx} / {frames.length - 1}
                                {currentFrame.actor !== null && (
                                    <span className="ml-3 text-ink-500">
                                        actor: player {currentFrame.actor} ({currentFrame.actor === 0 ? labelA : labelB})
                                    </span>
                                )}
                            </div>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={frames.length - 1}
                            value={frameIdx}
                            onChange={(e) => setFrameIdx(parseInt(e.target.value, 10))}
                            className="w-full"
                        />

                        <div className="rounded-lg border border-ink-800 bg-ink-950/40 p-4">
                            <components.Board
                                state={parsedState}
                                onAction={() => {}}
                                disabled
                            />
                        </div>

                        {(env === "hanabi" || env === "mini-hanabi") && parsedState && (
                            <HanabiDemoLog
                                frames={frames}
                                upToIdx={frameIdx}
                                state={parsedState as any}
                                playerNames={[labelA, labelB]}
                            />
                        )}
                        {!(env === "hanabi" || env === "mini-hanabi") && currentFrame.event && (
                            <div className="text-xs text-ink-400">
                                <span className="font-medium text-ink-200">
                                    {currentFrame.actor === 0 ? labelA : labelB}
                                </span>
                                <span className="font-mono"> · {String((currentFrame.event as any).kind ?? JSON.stringify(currentFrame.event))}</span>
                            </div>
                        )}
                    </div>
                )}

                <div className="flex gap-2">
                    <Link to={`/${env}/play`} className="btn-outline">
                        Back to play
                    </Link>
                </div>
            </div>
        </Layout>
    )
}

// Per-frame event narration for Hanabi/mini-hanabi demos so the user can see
// each move attributed to a specific policy.
function HanabiDemoLog({
    frames, upToIdx, state, playerNames,
}: {
    frames: { event: Record<string, unknown> | null; actor: number | null }[]
    upToIdx: number
    state: any
    playerNames: [string, string]
}) {
    const entries: LogEntry[] = []
    for (let i = 0; i <= Math.min(upToIdx, frames.length - 1); i++) {
        const f = frames[i]
        if (!f.event || f.actor == null) continue
        const ev = f.event as any
        const player = (f.actor === 0 ? 0 : 1) as 0 | 1
        if (ev.kind === "play" || ev.kind === "discard" || ev.kind === "hint_color" || ev.kind === "hint_rank" || ev.kind === "noop") {
            entries.push({
                player,
                kind: ev.kind,
                slot: ev.slot,
                color: ev.color,
                rank: ev.rank,
                bombed: ev.bombed,
                scored: ev.scored,
                revealed: ev.revealed,
            })
        }
    }
    return (
        <div className="rounded-lg border border-ink-800 bg-ink-950/40 p-4 max-h-72 overflow-y-auto">
            <div className="text-xs uppercase tracking-wider text-ink-500 mb-2">Action log</div>
            <GameLog entries={entries} state={state} playerNames={playerNames} />
        </div>
    )
}


// Inline form: pick a checkpoint zip + actor metadata, POST to /demo/upload,
// then it appears in the "My uploads" optgroup for both Policy A and Policy B.
function DemoUploadBox({ env, onUploaded }: { env: string; onUploaded: () => void }) {
    const [open, setOpen] = useState(false)
    const [file, setFile] = useState<File | null>(null)
    const [agentName, setAgentName] = useState("")
    const [actorType, setActorType] = useState<"mlp" | "s5" | "rnn">("mlp")
    const [archText, setArchText] = useState('{"FC_HIDDEN_DIM": 256, "ACTIVATION": "relu"}')
    const [ckptKey, setCkptKey] = useState("final_params")
    const [busy, setBusy] = useState(false)
    const [msg, setMsg] = useState<string | null>(null)

    const upload = async () => {
        if (!file) { setMsg("Choose a zipped saved_train_run/ first"); return }
        let archParams: Record<string, unknown> = {}
        try { archParams = JSON.parse(archText) }
        catch { setMsg("arch_params must be valid JSON"); return }
        setBusy(true); setMsg(null)
        try {
            await api.demoUpload(env, file, {
                agent_name: agentName || file.name.replace(/\.zip$/i, ""),
                actor_type: actorType,
                arch_params: archParams,
                ckpt_key: ckptKey,
            })
            setMsg("Uploaded. Available in 'My uploads'.")
            onUploaded()
        } catch (e) {
            setMsg(e instanceof Error ? e.message : String(e))
        } finally {
            setBusy(false)
        }
    }

    if (!open) {
        return (
            <div className="card-surface p-4 flex items-center justify-between">
                <div className="text-sm text-ink-400">
                    Want to demo your own policy?{" "}
                    <span className="text-ink-300">Upload a checkpoint</span>{" "}
                    (no evaluation runs, just stored for use in this page).
                </div>
                <button className="btn-outline text-sm" onClick={() => setOpen(true)}>Upload checkpoint</button>
            </div>
        )
    }

    return (
        <div className="card-surface p-6 space-y-3">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-ink-100">Upload checkpoint for demo</h3>
                <button className="text-xs text-ink-500 hover:text-ink-300" onClick={() => setOpen(false)}>cancel</button>
            </div>
            <p className="text-xs text-ink-500 leading-relaxed">
                Zip of an orbax{" "}
                <code className="text-ink-200 font-mono">saved_train_run/</code>
                {" "}directory. Stored under your session, no evaluation runs.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <label className="block">
                    <span className="text-xs uppercase tracking-wider text-ink-400">Checkpoint .zip</span>
                    <input type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                        className="mt-1 text-sm text-ink-300 file:mr-3 file:py-2 file:px-3 file:rounded file:border-0 file:bg-ink-800 file:text-ink-100" />
                </label>
                <label className="block">
                    <span className="text-xs uppercase tracking-wider text-ink-400">Agent name</span>
                    <input className="bench-input mt-1" value={agentName} onChange={(e) => setAgentName(e.target.value)} placeholder="optional" />
                </label>
                <label className="block">
                    <span className="text-xs uppercase tracking-wider text-ink-400">Actor type</span>
                    <select value={actorType} onChange={(e) => setActorType(e.target.value as any)} className="bench-input mt-1">
                        <option value="mlp">MLP</option>
                        <option value="s5">S5</option>
                        <option value="rnn">RNN</option>
                    </select>
                </label>
                <label className="block">
                    <span className="text-xs uppercase tracking-wider text-ink-400">Checkpoint key</span>
                    <input className="bench-input mt-1" value={ckptKey} onChange={(e) => setCkptKey(e.target.value)} />
                </label>
            </div>
            <label className="block">
                <span className="text-xs uppercase tracking-wider text-ink-400">Architecture parameters (JSON)</span>
                <textarea rows={2} className="bench-input mt-1 font-mono text-xs"
                    value={archText} onChange={(e) => setArchText(e.target.value)} />
            </label>
            <div className="flex items-center gap-3">
                <button className="btn-primary text-sm" onClick={upload} disabled={busy || !file}>
                    {busy ? "Uploading…" : "Upload"}
                </button>
                {msg && <span className="text-xs text-ink-400">{msg}</span>}
            </div>
        </div>
    )
}


function AgentSelect({
    label,
    options,
    value,
    onChange,
}: {
    label: string
    options: AgentOption[] | null
    value: string
    onChange: (id: string) => void
}) {
    const partners = (options ?? []).filter((o) => o.kind === "partner")
    const egos = (options ?? []).filter((o) => o.kind === "ego")
    const uploads = (options ?? []).filter((o) => o.kind === "upload")
    return (
        <label className="block">
            <span className="text-xs uppercase tracking-wider text-ink-400">{label}</span>
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                disabled={!options}
                className="mt-1 w-full bg-ink-900 border border-ink-700 rounded px-3 py-2 text-ink-100 text-sm"
            >
                {uploads.length > 0 && (
                    <optgroup label="My uploads">
                        {uploads.map((o) => (
                            <option key={o.id} value={o.id}>
                                {o.display_name}{o.tag ? ` (${o.tag})` : ""}
                            </option>
                        ))}
                    </optgroup>
                )}
                {partners.length > 0 && (
                    <optgroup label="Held-out partners">
                        {partners.map((o) => (
                            <option key={o.id} value={o.id}>
                                {o.display_name}{o.tag ? ` (${o.tag})` : ""}
                            </option>
                        ))}
                    </optgroup>
                )}
                {egos.length > 0 && (
                    <optgroup label="Builtin egos">
                        {egos.map((o) => (
                            <option key={o.id} value={o.id}>
                                {o.display_name}
                            </option>
                        ))}
                    </optgroup>
                )}
            </select>
        </label>
    )
}
