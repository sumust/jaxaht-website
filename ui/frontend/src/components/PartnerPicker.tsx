import type { PartnerInfo } from "../api/client"

type Props = {
    partners: PartnerInfo[]
    selected: string
    onSelect: (key: string) => void
}

const DIFFICULTY_PILLS: Record<PartnerInfo["difficulty"], string> = {
    easy: "bg-ink-800 text-ink-400",
    medium: "bg-hanabi-blue/20 text-hanabi-blue",
    hard: "bg-hanabi-yellow/20 text-hanabi-yellow",
    "human-like": "bg-hanabi-green/20 text-hanabi-green",
}

// Card-grid picker. Clicking a card selects it and expands its
// description. Groups partners by primary tag so the picker has structure.

const GROUP_ORDER = ["heuristic", "learned", "human-proxy", "specialist", "sequential", "greedy", "entitled", "walton-rivers", "reference", "pretrained"]
const GROUP_LABELS: Record<string, string> = {
    "heuristic": "Heuristic baselines",
    "walton-rivers": "Walton-Rivers heuristics",
    "learned": "Learned (RL)",
    "human-proxy": "Human proxy (BC)",
    "specialist": "Specialist heuristics",
    "sequential": "Sequential heuristics",
    "greedy": "Greedy heuristics",
    "entitled": "Entitled (cooperative)",
    "reference": "Reference implementations",
    "pretrained": "Pretrained (published)",
}

function groupPartners(partners: PartnerInfo[]): Array<{label: string; items: PartnerInfo[]}> {
    const groups = new Map<string, PartnerInfo[]>()
    for (const p of partners) {
        // pick the most specific (later-in-order) tag for grouping
        const primary = [...p.tags].reverse().find((t) => t in GROUP_LABELS) ?? p.tags[0] ?? "heuristic"
        if (!groups.has(primary)) groups.set(primary, [])
        groups.get(primary)!.push(p)
    }
    return GROUP_ORDER
        .filter((k) => groups.has(k))
        .map((k) => ({ label: GROUP_LABELS[k] ?? k, items: groups.get(k)! }))
}

export function PartnerPicker({ partners, selected, onSelect }: Props) {
    const groups = groupPartners(partners)
    return (
        <div className="space-y-6">
            {groups.map(({ label, items }) => (
                <div key={label}>
                    <h4 className="text-xs uppercase tracking-wider text-ink-500 mb-2">{label}</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {items.map((p) => renderCard(p, selected, onSelect))}
                    </div>
                </div>
            ))}
        </div>
    )
}

function renderCard(p: PartnerInfo, selected: string, onSelect: (k: string) => void) {
    const active = p.key === selected
    return (
        <button
            key={p.key}
            onClick={() => onSelect(p.key)}
            className={`
                text-left p-4 rounded-xl border transition-all
                ${active
                    ? "bg-ink-800/80 border-ink-500 ring-2 ring-ink-400 ring-offset-2 ring-offset-ink-950"
                    : "bg-ink-900/40 border-ink-800 hover:bg-ink-900 hover:border-ink-700"}
            `}
        >
            <div className="flex items-start justify-between gap-2 mb-2">
                <span className="font-medium text-ink-100">{p.display_name}</span>
                {p.difficulty && (
                    <span className={`pill ${DIFFICULTY_PILLS[p.difficulty] ?? "bg-ink-800 text-ink-400"}`}>
                        {p.difficulty}
                    </span>
                )}
            </div>
            <p className={`text-xs leading-relaxed ${active ? "text-ink-300" : "text-ink-500"}`}>
                {p.description}
            </p>
        </button>
    )
}

