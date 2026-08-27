import { useEffect, useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { useParams, useNavigate } from "react-router-dom"
import { Layout } from "../components/Layout"
import { ProgressBar } from "../components/ProgressBar"
import { Spinner } from "../components/Spinner"
import { PerPartnerChart } from "../components/PerPartnerChart"
import { api, ApiError, type BuiltinEgo, type HeldoutPartnerDescriptor, type JobStatus, type PerPartnerScore, type AggregateScore } from "../api/client"

// Submit flow:
//   1) pick a builtin ego (upload zip comes later, requires jax)
//   2) name the entry + choose episodes + seed
//   3) POST /submit → jobId
//   4) poll /submit/status until done
//   5) render per-partner breakdown + aggregate
//
// Polling uses a fixed 500 ms interval. The backend's progress payload
// is tiny so there's no bandwidth concern; real eval runs take minutes
// on Hanabi so the user watches a bar move.

type EvalResult = {
    per_partner: PerPartnerScore[]
    aggregate: AggregateScore
    num_episodes: number
    wall_clock_seconds: number
}

export function Submit() {
    const { env } = useParams<{ env: string }>()
    const navigate = useNavigate()
    if (!env) return null

    const [egos, setEgos] = useState<BuiltinEgo[] | null>(null)
    const [heldout, setHeldout] = useState<HeldoutPartnerDescriptor[] | null>(null)
    const [chosenEgo, setChosenEgo] = useState<string>("")
    const [agentName, setAgentName] = useState("")
    const [numEpisodes, setNumEpisodes] = useState(32)
    const [evalSeed, setEvalSeed] = useState(0)
    const [notes, setNotes] = useState("")
    const [error, setError] = useState<string | null>(null)

    // Upload mode (mutually exclusive with builtin ego).
    const [egoKind, setEgoKind] = useState<"builtin" | "upload">("upload")
    const [uploadFile, setUploadFile] = useState<File | null>(null)
    const [actorType, setActorType] = useState<"mlp" | "s5" | "rnn">("mlp")
    const [archParamsText, setArchParamsText] = useState('{"FC_HIDDEN_DIM": 256, "ACTIVATION": "relu"}')
    const [ckptKey, setCkptKey] = useState("final_params")
    const [ckptIdx, setCkptIdx] = useState(0)

    const [jobId, setJobId] = useState<string | null>(null)
    const [status, setStatus] = useState<JobStatus | null>(null)
    const [result, setResult] = useState<EvalResult | null>(null)

    useEffect(() => {
        api.egos(env).then((r) => {
            setEgos(r.egos)
            if (r.egos.length > 0) {
                setChosenEgo(r.egos[0].key)
                setAgentName(r.egos[0].display_name)
            }
        }).catch((e) => setError(e.message ?? String(e)))

        api.heldout(env).then((r) => setHeldout(r.partners))
            .catch((e) => setError(e.message ?? String(e)))
    }, [env])

    const submit = async () => {
        try {
            setError(null)
            setResult(null)
            setStatus(null)
            if (egoKind === "builtin") {
                const r = await api.submit(env, {
                    agent_name: agentName || undefined,
                    ego_kind: "builtin",
                    builtin_key: chosenEgo,
                    num_episodes: numEpisodes,
                    eval_seed: evalSeed,
                    notes: notes || null,
                })
                setJobId(r.job_id)
                return
            }
            // upload mode
            if (!uploadFile) { setError("Please choose a checkpoint .zip"); return }
            let archParams: Record<string, unknown> = {}
            try { archParams = JSON.parse(archParamsText) }
            catch { setError("arch_params must be valid JSON"); return }
            const payload = {
                agent_name: agentName || "Uploaded ego",
                ego_kind: "upload",
                actor_type: actorType,
                arch_params: archParams,
                ckpt_key: ckptKey,
                idx: ckptIdx,
                num_episodes: numEpisodes,
                eval_seed: evalSeed,
                notes: notes || null,
            }
            const fd = new FormData()
            fd.append("payload", JSON.stringify(payload))
            fd.append("checkpoint", uploadFile)
            const resp = await fetch(`/api/${env}/submit`, { method: "POST", body: fd })
            if (!resp.ok) {
                const t = await resp.text()
                throw new Error(`HTTP ${resp.status}: ${t.slice(0, 300)}`)
            }
            const data = await resp.json() as { job_id: string }
            setJobId(data.job_id)
        } catch (e) {
            setError(e instanceof ApiError ? e.message : String(e))
        }
    }

    // Polling loop
    useEffect(() => {
        if (!jobId) return
        let cancelled = false
        const poll = async () => {
            try {
                const s = await api.jobStatus(env, jobId)
                if (cancelled) return
                setStatus(s)
                if (s.status === "done") {
                    const res = await api.jobResult(env, jobId)
                    setResult(res as EvalResult)
                    return
                }
                if (s.status === "error") return
                setTimeout(poll, 500)
            } catch (e) {
                if (cancelled) return
                setError(e instanceof ApiError ? e.message : String(e))
            }
        }
        poll()
        return () => { cancelled = true }
    }, [jobId, env])

    const reset = () => {
        setJobId(null)
        setStatus(null)
        setResult(null)
    }

    return (
        <Layout
            title={`Submit an agent to ${envDisplayName(env)}`}
            subtitle="Scored against every held-out partner. Mean normalized return, bootstrap 95% CIs."
        >
            {error && (
                <div className="card-surface p-4 text-sm text-hanabi-red mb-6 animate-slide-up">
                    {error}
                </div>
            )}

            {!jobId && (
                <div className="grid grid-cols-12 gap-6">
                    <div className="col-span-12 lg:col-span-7 space-y-4">
                        <section className="card-surface p-6 space-y-4">
                            <h2 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                                Ego source
                            </h2>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setEgoKind("upload")}
                                    className={`btn-${egoKind === "upload" ? "primary" : "outline"} text-sm`}
                                >Upload checkpoint</button>
                                <button
                                    onClick={() => setEgoKind("builtin")}
                                    className={`btn-${egoKind === "builtin" ? "primary" : "outline"} text-sm`}
                                    disabled={!egos || egos.length === 0}
                                >Built-in ego{egos && egos.length === 0 ? " (none for this env)" : ""}</button>
                            </div>

                            {egoKind === "upload" && (
                                <div className="space-y-3">
                                    <p className="text-xs text-ink-500 leading-relaxed">
                                        Upload a zip of an orbax{" "}
                                        <code className="text-ink-200 font-mono">saved_train_run/</code>
                                        {" "}directory (the format that{" "}
                                        <code className="text-ink-200 font-mono">common.save_load_utils.save_train_run</code>
                                        {" "}produces in JaxAHT).
                                    </p>
                                    <Field label="Checkpoint file (.zip)">
                                        <input
                                            type="file"
                                            accept=".zip"
                                            onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                                            className="text-sm text-ink-300 file:mr-3 file:py-2 file:px-3 file:rounded file:border-0 file:bg-ink-800 file:text-ink-100 hover:file:bg-ink-700"
                                        />
                                    </Field>
                                    <div className="grid grid-cols-2 gap-3">
                                        <Field label="Actor type">
                                            <select
                                                value={actorType}
                                                onChange={(e) => setActorType(e.target.value as "mlp" | "s5" | "rnn")}
                                                className="bench-input"
                                            >
                                                <option value="mlp">MLP</option>
                                                <option value="s5">S5</option>
                                                <option value="rnn">RNN</option>
                                            </select>
                                        </Field>
                                        <Field label="Checkpoint key">
                                            <input className="bench-input" value={ckptKey} onChange={(e) => setCkptKey(e.target.value)} />
                                        </Field>
                                    </div>
                                    <Field label="Architecture parameters (JSON)">
                                        <textarea
                                            value={archParamsText}
                                            onChange={(e) => setArchParamsText(e.target.value)}
                                            rows={3}
                                            className="bench-input font-mono text-xs"
                                            placeholder='e.g. {"FC_HIDDEN_DIM": 256, "ACTIVATION": "relu"} or for S5 add S5_D_MODEL, S5_SSM_SIZE, S5_ACTOR_CRITIC_HIDDEN_DIM, FC_N_LAYERS'
                                        />
                                    </Field>
                                    <Field label="Checkpoint index (for multi-seed runs)">
                                        <input
                                            type="number" min={0}
                                            className="bench-input"
                                            value={ckptIdx}
                                            onChange={(e) => setCkptIdx(Math.max(0, Number(e.target.value) || 0))}
                                        />
                                    </Field>
                                </div>
                            )}

                            {egoKind === "builtin" && (
                                <div className="space-y-2">
                                    {!egos && <Spinner size="sm" label="Loading built-in egos…" />}
                                    {egos && egos.length > 0 && egos.map((ego) => (
                                        <button
                                            key={ego.key}
                                            onClick={() => {
                                                setChosenEgo(ego.key)
                                                setAgentName(ego.display_name)
                                            }}
                                            className={`
                                                w-full text-left p-3 rounded-lg border transition-all
                                                ${chosenEgo === ego.key
                                                    ? "bg-ink-800/80 border-ink-500 ring-2 ring-ink-400 ring-offset-2 ring-offset-ink-950"
                                                    : "bg-ink-900/40 border-ink-800 hover:bg-ink-900 hover:border-ink-700"}
                                            `}
                                        >
                                            <div className="font-medium text-ink-100">{ego.display_name}</div>
                                            <div className="text-xs text-ink-400 mt-1">{ego.description}</div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </section>

                        <section className="card-surface p-6 space-y-3">
                            <h2 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                                Submission metadata
                            </h2>
                            <Field label="Agent name (shown on leaderboard)">
                                <input
                                    className="bench-input"
                                    value={agentName}
                                    onChange={(e) => setAgentName(e.target.value)}
                                    maxLength={64}
                                />
                            </Field>
                            <Field label="Episodes per partner">
                                <input
                                    type="number"
                                    min={1}
                                    max={500}
                                    className="bench-input"
                                    value={numEpisodes}
                                    onChange={(e) => setNumEpisodes(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
                                />
                            </Field>
                            <Field label="Eval seed (for reproducibility)">
                                <input
                                    type="number"
                                    className="bench-input"
                                    value={evalSeed}
                                    onChange={(e) => setEvalSeed(Number(e.target.value) || 0)}
                                />
                            </Field>
                            <Field label="Notes (optional)">
                                <textarea
                                    className="bench-input min-h-[72px]"
                                    rows={3}
                                    value={notes}
                                    onChange={(e) => setNotes(e.target.value)}
                                    placeholder="Architecture, training algorithm, seeds. Shows up in the entry detail view."
                                />
                            </Field>
                        </section>

                        <div className="flex justify-end">
                            <button
                                className="btn-primary"
                                onClick={submit}
                                disabled={egoKind === "builtin" ? (!chosenEgo || !egos?.length) : !uploadFile}
                            >
                                Start evaluation
                            </button>
                        </div>
                    </div>

                    <aside className="col-span-12 lg:col-span-5 space-y-4">
                        <section className="card-surface p-6">
                            <h2 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-3">
                                Held-out suite
                                <span className="ml-2 text-ink-500 font-mono text-xs">v1</span>
                            </h2>
                            {!heldout && <Spinner size="sm" label="Loading partners…" />}
                            {heldout && (
                                <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                                    {heldout.map((p) => (
                                        <div key={p.key} className="border-l-2 border-ink-800 pl-3 py-1">
                                            <div className="flex items-center justify-between">
                                                <span className="text-sm text-ink-200">{p.display_name}</span>
                                                <span className="text-[10px] uppercase tracking-wider text-ink-500 font-mono">
                                                    {p.difficulty}
                                                </span>
                                            </div>
                                            <div className="text-xs text-ink-500 mt-0.5">{p.description}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {heldout && heldout.length === 0 && (
                                <p className="text-xs text-ink-500">No held-out partners for this env yet.</p>
                            )}
                        </section>
                    </aside>
                </div>
            )}

            {jobId && !result && status && (
                <section className="card-surface p-8 animate-slide-up space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm uppercase tracking-wider text-ink-500">
                                Job {jobId}
                            </div>
                            <div className="text-lg text-ink-200 mt-1">
                                {status.status === "pending" && "Queued…"}
                                {status.status === "running" && "Evaluating"}
                                {status.status === "error" && (
                                    <span className="text-hanabi-red">Failed</span>
                                )}
                            </div>
                        </div>
                        {status.status !== "error" && <Spinner size="md" />}
                    </div>
                    <ProgressBar
                        completed={status.progress.completed}
                        total={status.progress.total || (heldout?.length ?? 1)}
                        current={status.progress.current}
                    />
                    {status.error && (
                        <pre className="text-xs text-hanabi-red font-mono whitespace-pre-wrap mt-2 p-3 bg-ink-950 rounded border border-ink-800">
                            {status.error}
                        </pre>
                    )}
                </section>
            )}

            {result && (
                <section className="space-y-4 animate-fade-in">
                    <div className="card-surface p-8 text-center">
                        <div className="text-sm uppercase tracking-wider text-ink-500">
                            Aggregate (mean normalized return)
                        </div>
                        <div className="text-5xl font-bold text-ink-100 tabular-nums mt-3">
                            {result.aggregate.mean.toFixed(3)}
                        </div>
                        <div className="text-sm text-ink-400 mt-2">
                            95% CI&nbsp;[{result.aggregate.ci_low.toFixed(3)},&nbsp;
                            {result.aggregate.ci_high.toFixed(3)}]
                            &nbsp;·&nbsp;{result.aggregate.num_partners} partners
                            &nbsp;·&nbsp;{result.num_episodes} episodes each
                            &nbsp;·&nbsp;{result.wall_clock_seconds.toFixed(1)}s wall
                        </div>
                    </div>

                    <section className="card-surface p-6 space-y-3">
                        <h3 className="text-sm font-medium text-ink-300 uppercase tracking-wider">
                            Per-partner breakdown
                        </h3>
                        <PerPartnerChart scores={result.per_partner} />
                    </section>

                    <div className="flex justify-end gap-2">
                        <button
                            className="btn-outline"
                            onClick={() => navigate(`/${env}/leaderboard`)}
                        >
                            View leaderboard
                        </button>
                        <button className="btn-primary" onClick={reset}>
                            Submit another
                        </button>
                    </div>
                </section>
            )}
        </Layout>
    )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <label className="block text-sm space-y-1">
            <span className="text-xs uppercase tracking-wider text-ink-500">{label}</span>
            {children}
        </label>
    )
}
