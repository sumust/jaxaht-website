import { useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
import { Layout } from "../components/Layout"
import { api, ApiError } from "../api/client"

// Two-step consent flow ported from Johnny's human_data_collecting templates:
//   step 0 → Research Information Sheet (informed consent + study info)
//   step 1 → Welcome modal (game rules + an example) before play begins
//   step 2 → POST /study/start, navigate to play
type Step = "info" | "welcome"

export function Prolific() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [params] = useSearchParams()
    const navigate = useNavigate()
    const [step, setStep] = useState<Step>("info")
    const [consenting, setConsenting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [optInTraj, setOptInTraj] = useState(true)

    const prolific_pid = params.get("PROLIFIC_PID") ?? ""
    const study_id = params.get("STUDY_ID") ?? ""
    const prolific_session_id = params.get("SESSION_ID") ?? ""
    const has_prolific = Boolean(prolific_pid)

    const start = async () => {
        try {
            setConsenting(true)
            setError(null)
            const resp = await api.studyStart(env, {
                prolific_pid,
                study_id,
                prolific_session_id,
                data_source: has_prolific ? "prolific" : "test",
            })
            navigate(`/${env}/study?study=${encodeURIComponent(resp.study_id)}`)
        } catch (e) {
            setError(e instanceof ApiError ? e.message : String(e))
            setConsenting(false)
        }
    }

    return (
        <Layout
            title={`${envDisplayName(env)} study`}
            subtitle={has_prolific ? "Prolific worker session" : "Test session (no PID detected)"}
        >
            {step === "info" && (
                <InfoSheet
                    onAgree={() => setStep("welcome")}
                    optInTraj={optInTraj}
                    setOptInTraj={setOptInTraj}
                    prolific_pid={prolific_pid}
                    study_id={study_id}
                    prolific_session_id={prolific_session_id}
                />
            )}
            {step === "welcome" && (
                <Welcome
                    env={env}
                    onStart={start}
                    onBack={() => setStep("info")}
                    consenting={consenting}
                    error={error}
                />
            )}
        </Layout>
    )
}

function InfoSheet({
    onAgree, optInTraj, setOptInTraj,
    prolific_pid, study_id, prolific_session_id,
}: {
    onAgree: () => void
    optInTraj: boolean
    setOptInTraj: (v: boolean) => void
    prolific_pid: string
    study_id: string
    prolific_session_id: string
}) {
    return (
        <div className="card-surface p-8 space-y-5 max-w-3xl">
            <div>
                <h2 className="text-2xl font-semibold text-ink-100 mb-1">Research information sheet</h2>
                <p className="text-xs text-ink-500">Please read carefully before continuing.</p>
            </div>

            <section className="space-y-2 text-sm text-ink-400 leading-relaxed">
                <p>
                    <span className="text-ink-200 font-medium">Purpose.</span> You'll play a short series of
                    cooperative games against an AI partner. We're studying how humans coordinate with
                    AI agents in ad hoc settings — situations where neither side has trained against the
                    other before. Your gameplay helps us evaluate AI methods.
                </p>
                <p>
                    <span className="text-ink-200 font-medium">What you'll do.</span> 2 warmup games (uncounted)
                    to learn the controls, then 8 counted games. Each game is short (~30-60 seconds).
                    Total session is roughly 10-15 minutes.
                </p>
                <p>
                    <span className="text-ink-200 font-medium">Data we collect.</span> The sequence of moves you
                    make during each game (no personally identifying information beyond the Prolific ID
                    needed for payment). Trajectories are stored anonymously.
                </p>
                <p>
                    <span className="text-ink-200 font-medium">Voluntary.</span> You can stop at any time, but
                    payment requires finishing all counted games and submitting the completion code.
                </p>
                <p>
                    <span className="text-ink-200 font-medium">Contact.</span> Any concerns,
                    reach out via Prolific's messaging system.
                </p>
            </section>

            <label className="flex items-start gap-3 text-sm text-ink-300 border-t border-ink-800 pt-4 cursor-pointer">
                <input
                    type="checkbox"
                    checked={optInTraj}
                    onChange={(e) => setOptInTraj(e.target.checked)}
                    className="mt-0.5 accent-ut-orange"
                />
                <span>
                    I consent to having my gameplay trajectories stored anonymously for this research.
                </span>
            </label>

            {(prolific_pid || study_id) && (
                <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-3 text-xs font-mono text-ink-400 space-y-0.5">
                    <div>PROLIFIC_PID: <span className="text-ink-200">{prolific_pid || "(none)"}</span></div>
                    <div>STUDY_ID: <span className="text-ink-200">{study_id || "(none)"}</span></div>
                    <div>SESSION_ID: <span className="text-ink-200">{prolific_session_id || "(none)"}</span></div>
                </div>
            )}

            <div className="flex justify-end">
                <button onClick={onAgree} disabled={!optInTraj} className="btn-primary">
                    I agree, continue
                </button>
            </div>
        </div>
    )
}

function Welcome({
    env, onStart, onBack, consenting, error,
}: {
    env: string
    onStart: () => void
    onBack: () => void
    consenting: boolean
    error: string | null
}) {
    const isLBF = env === "lbf"
    return (
        <div className="card-surface p-8 space-y-5 max-w-3xl">
            <div>
                <h2 className="text-2xl font-semibold text-ink-100 mb-1">How to play {envDisplayName(env)}</h2>
                <p className="text-xs text-ink-500">A quick walkthrough before the warmup games.</p>
            </div>

            {isLBF && (
                <>
                    <section className="space-y-2 text-sm text-ink-400 leading-relaxed">
                        <p>
                            You're the <span className="text-hanabi-blue">blue player</span>, your partner is the{" "}
                            <span className="text-hanabi-yellow">yellow player</span>. The board has{" "}
                            <span className="text-ink-200">food</span> at various levels. Pick up food by stepping{" "}
                            <em>next to</em> it (not onto it) and pressing <kbd className="px-1.5 py-0.5 rounded bg-ink-800 text-ink-200 font-mono text-xs">space</kbd>.
                        </p>
                        <p>
                            Some food has a level higher than yours — you'll need to be adjacent to that food
                            <em> at the same time as your partner</em>, both pressing <kbd className="px-1.5 py-0.5 rounded bg-ink-800 text-ink-200 font-mono text-xs">space</kbd>,
                            for it to count as a cooperative load.
                        </p>
                    </section>
                    <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-4">
                        <img
                            src="/lbf-annotated.png"
                            alt="LBF board reference"
                            className="max-w-md mx-auto rounded"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none" }}
                        />
                        <p className="text-xs text-ink-500 mt-2 text-center">
                            Game board reference (food shows its required level).
                        </p>
                    </div>
                </>
            )}

            <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 text-sm text-ink-300 space-y-1">
                <div className="text-xs uppercase tracking-wider text-ink-500 mb-1">Controls</div>
                <div><kbd className="px-1.5 py-0.5 rounded bg-ink-800 text-ink-200 font-mono text-xs">W/A/S/D</kbd> or arrow keys — move</div>
                <div><kbd className="px-1.5 py-0.5 rounded bg-ink-800 text-ink-200 font-mono text-xs">Space</kbd> — interact / load / play</div>
                <div><kbd className="px-1.5 py-0.5 rounded bg-ink-800 text-ink-200 font-mono text-xs">Q</kbd> — wait / no-op</div>
            </div>

            <p className="text-xs text-ink-500">
                The first 2 games are <span className="text-ink-300">warmup</span> — they don't count and are just to learn the controls.
                After that, 8 counted games. When you finish, you'll get a completion code to submit on Prolific.
            </p>

            {error && <div className="text-sm text-red-400 font-mono">{error}</div>}

            <div className="flex items-center justify-between">
                <button onClick={onBack} className="btn-ghost text-sm">Back</button>
                <button onClick={onStart} disabled={consenting} className="btn-primary">
                    {consenting ? "Starting..." : "Start playing"}
                </button>
            </div>
        </div>
    )
}
