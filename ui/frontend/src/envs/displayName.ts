// env URL slug → human-readable display name. fallback to slug if unknown.
const ENV_DISPLAY: Record<string, string> = {
    "hanabi": "Hanabi",
    "mini-hanabi": "Mini-Hanabi",
    "lbf": "LBF",
    "overcooked-v1": "Overcooked",
}

export function envDisplayName(envSlug: string): string {
    return ENV_DISPLAY[envSlug] ?? envSlug
}
