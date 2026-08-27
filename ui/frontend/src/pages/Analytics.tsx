import { Link, useParams } from "react-router-dom"
import { Layout } from "../components/Layout"

// Placeholder page until the effort-analysis logic is generalized
// to read the unified TrajectoryStore. The preview below describes what
// will show up here.

export function Analytics() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    return (
        <Layout
            title={`${env} analytics`}
            subtitle="Post-hoc view of trajectories saved from play mode: scores, effort flags, episode replay."
        >
            <div className="card-surface p-8 space-y-6">
                <div>
                    <h2 className="text-xl font-semibold text-ink-100">
                        One dashboard across every env
                    </h2>
                    <p className="text-sm text-ink-400 mt-2 max-w-3xl leading-relaxed">
                        Every trajectory saved from a play session, whether Hanabi,
                        LBF, or Overcooked, lands in the same store. This page reads
                        that store and surfaces per-session stats, effort flags for
                        screening Prolific data, and step-through episode replay.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-sm">
                    <FeatureCard
                        title="Session stats"
                        body="Avg score, std dev, noop rate, action-loop detection. Sort by anything, filter by flag status."
                    />
                    <FeatureCard
                        title="Effort flags"
                        body="Catch workers who pressed the same key the whole time, or who scored zero for reasons that weren't the env. Threshold-based, tunable."
                    />
                    <FeatureCard
                        title="Episode replay"
                        body="Step through any saved game. Useful for debugging human/AI coordination failures directly."
                    />
                </div>

                <p className="text-xs text-ink-500 leading-relaxed max-w-3xl">
                    Lifts the{" "}
                    <code className="text-ink-300 font-mono">visualize.py</code>{" "}
                    LBF dashboard into a multi-env page. No more separate static HTML
                    per env; one live view that updates as humans play.
                </p>

                <div className="flex gap-2 pt-1">
                    <Link to={`/${env}/play`} className="btn-outline">
                        Back to play
                    </Link>
                </div>
            </div>
        </Layout>
    )
}

function FeatureCard({ title, body }: { title: string; body: string }) {
    return (
        <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-1.5">
            <div className="text-sm font-medium text-ink-200">{title}</div>
            <div className="text-xs text-ink-400 leading-relaxed">{body}</div>
        </div>
    )
}
