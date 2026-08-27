import { useState, useEffect } from "react"
import { envDisplayName } from "../envs/displayName"
import { useParams, useSearchParams } from "react-router-dom"
import { Layout } from "../components/Layout"

export function StudyComplete() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [params] = useSearchParams()
    const code = params.get("code") ?? ""
    const gamesPlayed = params.get("games") ?? ""
    // Base64-encoded Prolific completion URL (Johnny's anti-leakage convention:
    // workers see the encoded form in HTML source, only the decoded redirect target
    // fires on click). Falls back to the generic return URL when absent.
    const completionEncoded = params.get("completion") ?? ""
    const [copied, setCopied] = useState(false)
    const [returnHref, setReturnHref] = useState("https://app.prolific.com/submissions/complete")

    useEffect(() => {
        if (completionEncoded) {
            try { setReturnHref(atob(completionEncoded)) }
            catch { /* leave default */ }
        }
    }, [completionEncoded])

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        } catch { /* clipboard blocked, manual copy still works */ }
    }

    return (
        <Layout
            title={`${envDisplayName(env)} session complete`}
            subtitle="Thanks for playing. Your trajectories have been saved."
        >
            <div className="card-surface p-8 space-y-6 max-w-2xl text-center">
                <div className="text-5xl">🎉</div>
                <div>
                    <h2 className="text-2xl font-semibold text-ink-100 mb-1">Session complete</h2>
                    {gamesPlayed && (
                        <p className="text-sm text-ink-400">{gamesPlayed} games played</p>
                    )}
                </div>

                <div className="border-t border-ink-800 pt-6 space-y-3">
                    <p className="text-xs uppercase tracking-wider text-ink-500">Your completion code</p>
                    <div className="flex items-center justify-center gap-3">
                        <code className="px-4 py-3 rounded-lg bg-ink-900 border border-ink-700 text-xl font-mono text-ink-100 tracking-widest">
                            {code || "(no code)"}
                        </code>
                        <button onClick={copy} className="btn-outline" disabled={!code}>
                            {copied ? "Copied" : "Copy"}
                        </button>
                    </div>
                    <p className="text-xs text-ink-500 leading-relaxed">
                        Paste this code on Prolific to confirm your submission, or use the button below
                        to return directly.
                    </p>
                </div>

                {code && (
                    <div>
                        <a
                            href={returnHref}
                            className="btn-primary inline-block"
                            target="_top"
                            rel="noreferrer"
                        >
                            Return to Prolific to submit
                        </a>
                    </div>
                )}
            </div>
        </Layout>
    )
}
