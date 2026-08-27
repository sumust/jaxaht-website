import { useCallback, useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { Layout } from "../components/Layout"
import { Spinner } from "../components/Spinner"
import { api, type EnvInfo } from "../api/client"

export function Home() {
    const [envs, setEnvs] = useState<EnvInfo[] | null>(null)
    const [error, setError] = useState<string | null>(null)

    const load = useCallback(() => {
        setError(null)
        setEnvs(null)
        api.envs()
            .then((r) => setEnvs(r.envs))
            .catch((e) => setError(e.message ?? String(e)))
    }, [])

    useEffect(() => { load() }, [load])

    return (
        <Layout
            title="JaxAHT"
            subtitle="Live deployment of the JaxAHT codebase for ad hoc teamwork research."
        >
            <section className="mb-8 card-surface p-6 bg-gradient-to-br from-ut-orange/5 via-ink-900/0 to-ink-900/0">
                <h2 className="text-base font-semibold text-ink-100 mb-2">What this is for</h2>
                <p className="text-sm text-ink-400 leading-relaxed max-w-3xl">
                    JaxAHT is a JAX-based codebase for ad hoc teamwork. This Space exposes the
                    four supported environments and their held-out partner sets to anyone who
                    wants to interact with them: play in the browser against a partner, run a
                    policy demo between two agents of your choice, submit an ego checkpoint to be
                    evaluated against the full held-out suite, browse leaderboard results from
                    other submissions, or run a Prolific study for human data collection.
                </p>
            </section>
            {error && (
                <div className="card-surface p-4 mb-6 flex items-start justify-between gap-4">
                    <div className="space-y-1">
                        <div className="text-sm font-medium text-hanabi-red">
                            Could not reach the backend.
                        </div>
                        <div className="text-xs text-ink-400 font-mono whitespace-pre-wrap">
                            {error}
                        </div>
                        <div className="text-xs text-ink-500">
                            Start it: <code className="text-ink-300 bg-ink-950 px-1 py-0.5 rounded">bash ui/dev.sh</code>
                            &nbsp; · &nbsp; Logs: <code className="text-ink-300 bg-ink-950 px-1 py-0.5 rounded">tail -f /tmp/ui.backend.log</code>
                        </div>
                    </div>
                    <button className="btn-outline shrink-0" onClick={load}>
                        Retry
                    </button>
                </div>
            )}

            {envs === null && !error && (
                <div className="py-10 flex justify-center">
                    <Spinner size="md" label="Loading environments…" />
                </div>
            )}

            {envs && envs.length === 0 && (
                <div className="text-sm text-ink-500">No environments registered.</div>
            )}

            {envs && envs.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    {envs.map((env) => <EnvCard key={env.env_name} env={env} />)}
                </div>
            )}

            <section className="mt-14 card-surface p-8">
                <h2 className="text-sm font-medium text-ink-300 uppercase tracking-wider mb-4">
                    What you can do here
                </h2>
                <ul className="space-y-2.5 text-sm text-ink-400 leading-relaxed">
                    <li>
                        <span className="text-ink-200 font-medium">Play</span> a round in the browser
                        against any held-out partner for the environment.
                    </li>
                    <li>
                        <span className="text-ink-200 font-medium">Policy demo</span>: pick any two
                        agents (held-out partners, built-in egos, or your uploaded checkpoints), run
                        an episode, step through the frames.
                    </li>
                    <li>
                        <span className="text-ink-200 font-medium">Submit a checkpoint</span> for
                        full evaluation against the held-out suite. The result is a leaderboard
                        entry with per-partner scores and bootstrap confidence intervals.
                    </li>
                    <li>
                        <span className="text-ink-200 font-medium">Leaderboard</span> per environment,
                        with per-partner detail on each entry and a CSV export.
                    </li>
                    <li>
                        <span className="text-ink-200 font-medium">Prolific study</span>: multi-game
                        human data collection session that produces a completion code (LBF only at the
                        moment).
                    </li>
                </ul>
            </section>
        </Layout>
    )
}

const ACCENT_TINT: Record<string, string> = {
    blue:   "from-hanabi-blue/10   via-ink-900/60     to-ink-900/80",
    green:  "from-hanabi-green/10  via-ink-900/60     to-ink-900/80",
    yellow: "from-hanabi-yellow/10 via-ink-900/60     to-ink-900/80",
    red:    "from-hanabi-red/10    via-ink-900/60     to-ink-900/80",
}
const ACCENT_DOT: Record<string, string> = {
    blue:   "bg-hanabi-blue",
    green:  "bg-hanabi-green",
    yellow: "bg-hanabi-yellow",
    red:    "bg-hanabi-red",
}

function EnvCard({ env }: { env: EnvInfo }) {
    const tint = ACCENT_TINT[env.accent] ?? ACCENT_TINT.blue
    const dot = ACCENT_DOT[env.accent] ?? ACCENT_DOT.blue
    const statEntries = Object.entries(env.stats)

    return (
        <article className={`
            relative rounded-xl border border-ink-800 overflow-hidden
            bg-gradient-to-br ${tint}
            shadow-xl shadow-black/20 transition-all duration-200
            flex flex-col h-full
            ${env.ready ? "hover:border-ink-700" : ""}
        `}>
            <div className="p-6 space-y-4 flex flex-col flex-grow">
                <div className="flex items-start gap-3">
                    <span className={`w-2 h-2 rounded-full mt-2.5 shrink-0 ${dot}`} />
                    <div className="min-w-0">
                        <h3 className="text-xl font-semibold text-ink-100 tracking-tight">
                            {env.display_name}
                        </h3>
                        {env.overview && (
                            <p className="text-sm text-ink-400 mt-1.5 text-balance leading-relaxed">
                                {env.overview}
                            </p>
                        )}
                    </div>
                </div>

                {(statEntries.length > 0 || env.num_partners > 0) && (
                    <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs pl-5">
                        {statEntries.map(([k, v]) => (
                            <span key={k} className="text-ink-500">
                                <span className="uppercase tracking-wider">{k}</span>{" "}
                                <span className="text-ink-200 font-mono tabular-nums">{v}</span>
                            </span>
                        ))}
                        {env.num_partners > 0 && (
                            <span className="text-ink-500">
                                <span className="uppercase tracking-wider">held-out</span>{" "}
                                <span className="text-ink-200 font-mono tabular-nums">
                                    {env.num_partners}
                                </span>
                            </span>
                        )}
                    </div>
                )}

                <div className="flex flex-wrap gap-2 pt-2 mt-auto">
                    {env.ready && env.modes.includes("play") && (
                        <Link to={`/${env.env_name}/play`} className="btn-primary">
                            Play
                        </Link>
                    )}
                    {env.ready && env.modes.includes("submit") && (
                        <Link to={`/${env.env_name}/submit`} className="btn-outline">
                            Submit
                        </Link>
                    )}
                    {env.ready && env.modes.includes("leaderboard") && (
                        <Link to={`/${env.env_name}/leaderboard`} className="btn-outline">
                            Leaderboard
                        </Link>
                    )}
                    {env.ready && env.modes.includes("demo") && (
                        <Link to={`/${env.env_name}/demo`} className="btn-outline">
                            Policy demo
                        </Link>
                    )}
                    {!env.ready && (
                        <button className="btn-outline" disabled>
                            Not hooked up yet
                        </button>
                    )}
                </div>
            </div>
        </article>
    )
}
