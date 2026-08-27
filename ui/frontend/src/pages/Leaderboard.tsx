import { useEffect, useState } from "react"
import { envDisplayName } from "../envs/displayName"
import { Link, useParams } from "react-router-dom"
import { Layout } from "../components/Layout"
import { PerPartnerChart } from "../components/PerPartnerChart"
import { Spinner } from "../components/Spinner"
import { api, type LeaderboardEntry } from "../api/client"

// Leaderboard page:
//   - Sortable table of all entries for the chosen (env, version)
//   - Click row to expand per-partner breakdown
//   - CSV / LaTeX export of the current table
//   - Version selector (empty today, v1 only; ready for v2+)
//
// Sort defaults to aggregate_score descending. Click a column header
// to flip.

type SortKey = "aggregate_score" | "agent_name" | "created_at" | "num_episodes"

export function Leaderboard() {
    const { env } = useParams<{ env: string }>()
    if (!env) return null

    const [entries, setEntries] = useState<LeaderboardEntry[] | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [expanded, setExpanded] = useState<string | null>(null)
    const [sortKey, setSortKey] = useState<SortKey>("aggregate_score")
    const [sortDir, setSortDir] = useState<"asc" | "desc">("desc")
    const [version] = useState("v1")
    const [mode, setMode] = useState<"normalized" | "raw">("normalized")

    useEffect(() => {
        api.leaderboard(env, version)
            .then((r) => setEntries(r.entries))
            .catch((e) => setError(e.message ?? String(e)))
    }, [env, version])

    const sorted = entries ? [...entries].sort((a, b) => {
        const key = sortKey
        const av = key === "agent_name" ? a.agent_name : (a as any)[key] ?? 0
        const bv = key === "agent_name" ? b.agent_name : (b as any)[key] ?? 0
        const cmp = av < bv ? -1 : av > bv ? 1 : 0
        return sortDir === "asc" ? cmp : -cmp
    }) : null

    const toggleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir((d) => d === "asc" ? "desc" : "asc")
        } else {
            setSortKey(key)
            setSortDir("desc")
        }
    }

    const exportCsv = () => {
        if (!entries) return
        const head = ["rank", "agent_name", "aggregate_score", "ci_low", "ci_high", "num_episodes", "eval_seed", "env", "version", "created_at"]
        const rows = sorted!.map((e, i) => [
            i + 1,
            csvCell(e.agent_name),
            e.aggregate_score.toFixed(4),
            e.aggregate.ci_low.toFixed(4),
            e.aggregate.ci_high.toFixed(4),
            e.num_episodes,
            e.eval_seed,
            e.env,
            e.version,
            new Date(e.created_at * 1000).toISOString(),
        ])
        const csv = [head, ...rows].map((row) => row.join(",")).join("\n")
        const blob = new Blob([csv], { type: "text/csv" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `${env}_leaderboard_${version}.csv`
        a.click()
        URL.revokeObjectURL(url)
    }

    const exportLatex = () => {
        if (!entries) return
        const body = sorted!.map((e, i) =>
            `${i + 1} & ${latexEscape(e.agent_name)} & ${e.aggregate_score.toFixed(3)} & [${e.aggregate.ci_low.toFixed(3)}, ${e.aggregate.ci_high.toFixed(3)}] & ${e.num_episodes} \\\\`
        ).join("\n")
        const tex =
`\\begin{table}
  \\centering
  \\caption{${env} leaderboard (${version}). Mean normalized return with 95\\% bootstrap CI across ${entries[0]?.aggregate.num_partners ?? "?"} held-out partners.}
  \\begin{tabular}{r l c c r}
    \\toprule
    Rank & Agent & Score & 95\\% CI & Episodes \\\\
    \\midrule
${body}
    \\bottomrule
  \\end{tabular}
\\end{table}`
        const blob = new Blob([tex], { type: "text/plain" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `${env}_leaderboard_${version}.tex`
        a.click()
        URL.revokeObjectURL(url)
    }

    return (
        <Layout
            title={`${envDisplayName(env)} leaderboard`}
            subtitle="Mean normalized return across the full held-out suite. Bootstrap 95% CIs."
            right={
                <div className="flex items-center gap-2">
                    <Link to={`/${env}/submit`} className="btn-primary">
                        Submit an agent
                    </Link>
                </div>
            }
        >
            {error && (
                <div className="card-surface p-4 text-sm text-hanabi-red mb-6">{error}</div>
            )}

            {!entries && !error && <Spinner size="md" label="Loading leaderboard…" />}

            {entries && entries.length === 0 && (
                <section className="card-surface p-8 text-center">
                    <p className="text-ink-400">No submissions yet for this env + version.</p>
                    <Link to={`/${env}/submit`} className="btn-primary mt-4 inline-flex">
                        Be the first
                    </Link>
                </section>
            )}

            {sorted && sorted.length > 0 && (
                <section className="card-surface overflow-hidden">
                    <div className="p-4 border-b border-ink-800 flex items-center justify-between">
                        <div className="flex items-center gap-3 text-xs">
                            <span className="uppercase tracking-wider text-ink-500">Score mode</span>
                            <div className="flex gap-1 bg-ink-900 rounded-md p-0.5 border border-ink-800">
                                <button
                                    className={`px-3 py-1 rounded ${mode === "normalized" ? "bg-ink-800 text-ink-100" : "text-ink-400"}`}
                                    onClick={() => setMode("normalized")}
                                >
                                    normalized
                                </button>
                                <button
                                    className={`px-3 py-1 rounded ${mode === "raw" ? "bg-ink-800 text-ink-100" : "text-ink-400"}`}
                                    onClick={() => setMode("raw")}
                                >
                                    raw return
                                </button>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <button className="btn-ghost text-xs" onClick={exportCsv}>CSV</button>
                            <button className="btn-ghost text-xs" onClick={exportLatex}>LaTeX</button>
                        </div>
                    </div>
                    <table className="w-full text-sm">
                        <thead className="text-xs uppercase tracking-wider text-ink-500 border-b border-ink-800">
                            <tr>
                                <th className="px-4 py-3 text-left w-16">Rank</th>
                                <SortableTh onClick={() => toggleSort("agent_name")} active={sortKey === "agent_name"} dir={sortDir}>
                                    Agent
                                </SortableTh>
                                <SortableTh onClick={() => toggleSort("aggregate_score")} active={sortKey === "aggregate_score"} dir={sortDir} align="right">
                                    Score
                                </SortableTh>
                                <th className="px-4 py-3 text-right">95% CI</th>
                                <SortableTh onClick={() => toggleSort("num_episodes")} active={sortKey === "num_episodes"} dir={sortDir} align="right">
                                    Ep / partner
                                </SortableTh>
                                <SortableTh onClick={() => toggleSort("created_at")} active={sortKey === "created_at"} dir={sortDir} align="right">
                                    Submitted
                                </SortableTh>
                                <th className="px-4 py-3 w-10" />
                            </tr>
                        </thead>
                        <tbody>
                            {sorted.map((entry, i) => (
                                <LeaderboardRow
                                    key={entry.id}
                                    entry={entry}
                                    rank={i + 1}
                                    expanded={expanded === entry.id}
                                    onToggle={() => setExpanded(expanded === entry.id ? null : entry.id)}
                                    chartMode={mode}
                                />
                            ))}
                        </tbody>
                    </table>
                </section>
            )}
        </Layout>
    )
}

function LeaderboardRow({
    entry, rank, expanded, onToggle, chartMode,
}: {
    entry: LeaderboardEntry
    rank: number
    expanded: boolean
    onToggle: () => void
    chartMode: "normalized" | "raw"
}) {
    const submittedAgo = humanRelTime(entry.created_at)
    return (
        <>
            <tr
                onClick={onToggle}
                className={`
                    border-b border-ink-800/50 hover:bg-ink-900/40 cursor-pointer
                    ${expanded ? "bg-ink-900/40" : ""}
                `}
            >
                <td className="px-4 py-3">
                    <span className={`font-mono tabular-nums ${rank === 1 ? "text-hanabi-yellow" : "text-ink-400"}`}>
                        {rank}
                    </span>
                </td>
                <td className="px-4 py-3">
                    <div className="font-medium text-ink-100">{entry.agent_name}</div>
                    <div className="text-[10px] text-ink-500 font-mono mt-0.5">
                        {entry.ego_kind}
                        {entry.builtin_key && ` · ${entry.builtin_key}`}
                        {entry.checkpoint_sha256 && ` · ${entry.checkpoint_sha256.slice(0, 8)}`}
                    </div>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-ink-100 text-base">
                    {entry.aggregate_score.toFixed(3)}
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-ink-400 text-xs">
                    [{entry.aggregate.ci_low.toFixed(3)}, {entry.aggregate.ci_high.toFixed(3)}]
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-ink-400">
                    {entry.num_episodes}
                </td>
                <td className="px-4 py-3 text-right text-ink-500 text-xs">
                    {submittedAgo}
                </td>
                <td className="px-4 py-3 text-ink-500">
                    {expanded ? "▾" : "▸"}
                </td>
            </tr>
            {expanded && (
                <tr className="border-b border-ink-800">
                    <td colSpan={7} className="px-6 py-4 bg-ink-950/60 animate-slide-up">
                        <div className="space-y-4">
                            {entry.notes && (
                                <div className="text-xs text-ink-400 italic">
                                    {entry.notes}
                                </div>
                            )}
                            <PerPartnerChart scores={entry.per_partner} mode={chartMode} />
                            <div className="flex gap-6 text-xs text-ink-500 font-mono pt-2">
                                <span>seed: {entry.eval_seed}</span>
                                <span>version: {entry.version}</span>
                                <span>{entry.aggregate.num_partners} partners</span>
                                <span>{entry.wall_clock_seconds.toFixed(1)}s wall</span>
                            </div>
                        </div>
                    </td>
                </tr>
            )}
        </>
    )
}

function SortableTh({
    children, onClick, active, dir, align = "left",
}: {
    children: React.ReactNode
    onClick: () => void
    active: boolean
    dir: "asc" | "desc"
    align?: "left" | "right"
}) {
    return (
        <th
            onClick={onClick}
            className={`
                px-4 py-3 cursor-pointer select-none
                ${align === "right" ? "text-right" : "text-left"}
                ${active ? "text-ink-200" : "hover:text-ink-300"}
            `}
        >
            <span>{children}</span>
            {active && <span className="ml-1">{dir === "asc" ? "▲" : "▼"}</span>}
        </th>
    )
}

function humanRelTime(ts: number): string {
    const sec = Math.max(0, Date.now() / 1000 - ts)
    if (sec < 60) return "just now"
    if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
    if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
    return `${Math.floor(sec / 86400)}d ago`
}

function csvCell(s: string): string {
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`
    return s
}

function latexEscape(s: string): string {
    return s.replace(/([&%$#_{}~^\\])/g, "\\$1")
}
