import { Layout } from "../components/Layout"

export function About() {
    return (
        <Layout title="About" subtitle="Background, citation, and how to plug in.">
            <div className="space-y-6 max-w-3xl">
                <section className="card-surface p-6 space-y-3">
                    <h2 className="text-lg font-semibold text-ink-100">What this is</h2>
                    <p className="text-sm text-ink-400 leading-relaxed">
                        A live deployment of the JaxAHT codebase. Four cooperative environments are
                        supported: Hanabi, Mini-Hanabi, Level-Based Foraging, and Overcooked-v1.
                        For each environment, you can play a game in your browser against any
                        held-out partner, run a policy demo between two agents of your choice,
                        upload an ego checkpoint to be evaluated against the held-out suite,
                        and browse the resulting leaderboard. Also includes Prolific studies for
                        human data collection in LBF.
                    </p>
                </section>

                <section className="card-surface p-6 space-y-3">
                    <h2 className="text-lg font-semibold text-ink-100">Submitting a checkpoint</h2>
                    <p className="text-sm text-ink-400 leading-relaxed">
                        Checkpoints should be a zip of an orbax{" "}
                        <code className="text-ink-200 font-mono text-xs">saved_train_run/</code>
                        {" "}directory (the format that{" "}
                        <code className="text-ink-200 font-mono text-xs">common.save_load_utils.save_train_run</code>
                        {" "}produces in JaxAHT). Submission accepts MLP, S5, and RNN actor types,
                        plus their architecture parameters. Two paths:
                    </p>
                    <ul className="text-sm text-ink-400 leading-relaxed list-disc pl-6 space-y-1">
                        <li>
                            <span className="text-ink-200">Submit page</span>: upload your zip + actor metadata,
                            we evaluate it server-side against every held-out partner for that environment and write the entry.
                        </li>
                        <li>
                            <span className="text-ink-200">Policy demo page</span>: upload a checkpoint for visualization only
                            (no evaluation), then use it as one of the two agents in a demo episode.
                        </li>
                    </ul>
                </section>

                <section className="card-surface p-6 space-y-3">
                    <h2 className="text-lg font-semibold text-ink-100">Citation</h2>
                    <pre className="text-xs font-mono bg-ink-900 border border-ink-800 rounded p-3 text-ink-200 overflow-x-auto">
{`@misc{jaxaht2026,
  title={JaxAHT: A JAX-Based Library for Ad Hoc Teamwork},
  year={2026},
  howpublished={\\url{https://github.com/LARG/jax-aht}},
}`}
                    </pre>
                </section>

                <section className="card-surface p-6 space-y-2">
                    <h2 className="text-lg font-semibold text-ink-100">Documentation</h2>
                    <p className="text-sm text-ink-400">
                        Full docs coming soon.
                    </p>
                </section>
            </div>
        </Layout>
    )
}
