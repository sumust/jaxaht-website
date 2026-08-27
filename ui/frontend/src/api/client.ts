// Typed fetch wrappers over the Flask backend. Each function mirrors
// a Pydantic schema in backend/schemas.py. Keep these in sync: if you
// add a route on the backend, add a wrapper here.

export type PartnerInfo = {
    key: string
    display_name: string
    difficulty: "easy" | "medium" | "hard" | "human-like"
    description: string
    tags: string[]
}

export type EnvInfo = {
    env_name: string
    display_name: string
    modes: string[]
    default_partner_key: string
    overview: string
    stats: Record<string, string>
    accent: "blue" | "yellow" | "green" | "red" | string
    ready: boolean
    num_partners: number
}

export type NewGameResponse = {
    session_id: string
    state: Record<string, unknown>
    score: Record<string, unknown>
}

export type StepResponse = {
    state: Record<string, unknown>
    score: Record<string, unknown>
    reward: number
    done: boolean
    info: Record<string, unknown>
    partner_acted: boolean
    events: Record<string, unknown>[]
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
    let resp: Response
    try {
        resp = await fetch(url, {
            ...init,
            headers: {
                "Content-Type": "application/json",
                ...(init?.headers ?? {}),
            },
        })
    } catch (cause) {
        // Pure network failure: DNS down, CORS block, Vite proxy has
        // no upstream to talk to, etc. fetch throws a TypeError with
        // "Failed to fetch" which is cryptic, so we unwrap it.
        throw new ApiError(
            "network_error",
            "Could not reach the backend. Start it with `bash ui/dev.sh`.",
            0,
        )
    }

    if (!resp.ok) {
        // The backend (when reached) always returns JSON errors.
        // If we get non-JSON here, it's usually Vite's own 502 HTML
        // surfaced because Flask isn't listening on :5174.
        let parsed: { error?: string; detail?: string } | null = null
        try { parsed = await resp.json() } catch { /* non-JSON response */ }
        if (parsed) {
            throw new ApiError(
                parsed.error ?? `HTTP ${resp.status}`,
                parsed.detail ?? resp.statusText,
                resp.status,
            )
        }
        throw new ApiError(
            `HTTP ${resp.status}`,
            resp.status === 502
                ? "Vite proxy got a 502: the Flask backend on :5174 isn't responding."
                : resp.statusText || "non-JSON error response",
            resp.status,
        )
    }
    return resp.json() as Promise<T>
}

export class ApiError extends Error {
    constructor(
        public readonly code: string,
        public readonly detail: string,
        public readonly status: number,
    ) {
        super(detail ? `${code}: ${detail}` : code)
    }
}

export type BuiltinEgo = {
    key: string
    display_name: string
    description: string
    tags: string[]
}

export type UploadedPartnerInfo = {
    checkpoint_id: string
    agent_name: string
    actor_type: string | null
    aggregate_score: number | null
    created_at: number | null
    checkpoint_sha256: string | null
}

export type HeldoutPartnerDescriptor = {
    key: string
    display_name: string
    difficulty: string
    description: string
    tags: string[]
    normalize_bounds: [number, number] | null
}

export type PerPartnerScore = {
    key: string
    display_name: string
    mean: number
    std: number
    ci_low: number
    ci_high: number
    normalized_mean: number
    normalized_ci_low: number
    normalized_ci_high: number
    n_episodes: number
    mean_steps: number
}

export type AggregateScore = {
    method: string
    mean: number
    ci_low: number
    ci_high: number
    num_partners: number
}

export type LeaderboardEntry = {
    id: string
    env: string
    version: string
    agent_name: string
    aggregate_score: number
    aggregate: AggregateScore
    per_partner: PerPartnerScore[]
    ego_kind: "builtin" | "upload"
    builtin_key: string | null
    checkpoint_sha256: string | null
    num_episodes: number
    eval_seed: number
    notes: string | null
    created_at: number
    wall_clock_seconds: number
}

export type DemoFrame = {
    state: Record<string, unknown>
    score: Record<string, unknown>
    step: number
    actor: number | null
    action: number | null
    reward: number
    event: Record<string, unknown> | null
}

export type DemoResponse = {
    frames: DemoFrame[]
    total_reward: number
    final_score: Record<string, unknown>
    done: boolean
    num_frames: number
    agent_a_id: string
    agent_b_id: string
}

export type StudyStateResponse = {
    study_id: string
    current_game_index: number
    total_games: number
    num_warmup: number
    is_warmup: boolean
    session_id: string
    state: Record<string, unknown>
    score: Record<string, unknown>
    partner_key: string
}

export type StudyStepResponse = StudyStateResponse & {
    session_complete: boolean
    completion_code: string | null
    game_just_advanced: boolean
    prev_game_index: number | null
}

export type JobStatus = {
    id: string
    kind: string
    env: string
    status: "pending" | "running" | "done" | "error"
    progress: { completed: number; total: number; current: string | null }
    error: string | null
    created_at: number
    updated_at: number
    started_at: number | null
    finished_at: number | null
}

export const api = {
    envs: () => jsonFetch<{ envs: EnvInfo[] }>("/api/envs"),

    partners: (env: string) =>
        jsonFetch<{ partners: PartnerInfo[] }>(`/api/${env}/partners`),

    heldout: (env: string, version = "v1") =>
        jsonFetch<{ env: string; version: string; partners: HeldoutPartnerDescriptor[] }>(
            `/api/${env}/heldout?version=${encodeURIComponent(version)}`,
        ),

    heldoutVersions: (env: string) =>
        jsonFetch<{ versions: string[] }>(`/api/${env}/heldout/versions`),

    egos: (env: string) =>
        jsonFetch<{ egos: BuiltinEgo[] }>(`/api/${env}/egos`),

    uploadedPartners: (env: string) =>
        jsonFetch<{ uploaded: UploadedPartnerInfo[] }>(`/api/${env}/uploaded_partners`),

    demoUpload: async (env: string, file: File, meta: {
        agent_name: string
        actor_type: "mlp" | "s5" | "rnn"
        arch_params: Record<string, unknown>
        ckpt_key?: string
        idx?: number
    }) => {
        const fd = new FormData()
        fd.append("payload", JSON.stringify(meta))
        fd.append("checkpoint", file)
        const r = await fetch(`/api/${env}/demo/upload`, { method: "POST", body: fd })
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`)
        return (await r.json()) as { checkpoint_id: string; entry_id: string; ok: boolean }
    },

    newGame: (env: string, partnerKey: string, seed?: number, envKwargs?: Record<string, unknown>) =>
        jsonFetch<NewGameResponse>(`/api/${env}/play/new`, {
            method: "POST",
            body: JSON.stringify({
                partner_key: partnerKey,
                seed: seed ?? null,
                env_kwargs: envKwargs ?? {},
            }),
        }),

    step: (env: string, sessionId: string, action: Record<string, unknown>) =>
        jsonFetch<StepResponse>(`/api/${env}/play/step`, {
            method: "POST",
            body: JSON.stringify({ session_id: sessionId, action }),
        }),

    saveTrajectory: (env: string, sessionId: string, agentName: string) =>
        jsonFetch<{ trajectory_id: string }>(`/api/${env}/play/save`, {
            method: "POST",
            body: JSON.stringify({ session_id: sessionId, agent_name: agentName }),
        }),

    demo: (env: string, payload: {
        agent_a_id: string
        agent_b_id: string
        seed?: number
        max_steps?: number
    }) =>
        jsonFetch<DemoResponse>(`/api/${env}/play/demo`, {
            method: "POST",
            body: JSON.stringify(payload),
        }),

    studyStart: (env: string, prolific: {
        prolific_pid?: string
        study_id?: string
        prolific_session_id?: string
        data_source?: string
    }) =>
        jsonFetch<StudyStateResponse>(`/api/${env}/study/start`, {
            method: "POST",
            body: JSON.stringify(prolific),
        }),

    studyStep: (env: string, studyId: string, action: Record<string, unknown>) =>
        jsonFetch<StudyStepResponse>(`/api/${env}/study/step`, {
            method: "POST",
            body: JSON.stringify({ study_id: studyId, action }),
        }),

    studySave: (env: string, studyId: string, agentName: string) =>
        jsonFetch<{ trajectory_ids: string[]; count: number }>(`/api/${env}/study/save`, {
            method: "POST",
            body: JSON.stringify({ study_id: studyId, agent_name: agentName }),
        }),

    submit: (env: string, payload: {
        agent_name?: string
        version?: string
        num_episodes?: number
        eval_seed?: number
        ego_kind?: "builtin" | "upload"
        builtin_key?: string | null
        notes?: string | null
    }) =>
        jsonFetch<{ job_id: string }>(`/api/${env}/submit`, {
            method: "POST",
            body: JSON.stringify(payload),
        }),

    jobStatus: (env: string, jobId: string) =>
        jsonFetch<JobStatus>(`/api/${env}/submit/status/${jobId}`),

    jobResult: (env: string, jobId: string) =>
        jsonFetch<{
            entry_id: string
            per_partner: PerPartnerScore[]
            aggregate: AggregateScore
            num_episodes: number
            wall_clock_seconds: number
        }>(`/api/${env}/submit/result/${jobId}`),

    leaderboard: (env: string, version = "v1") =>
        jsonFetch<{ env: string; version: string; entries: LeaderboardEntry[] }>(
            `/api/${env}/leaderboard?version=${encodeURIComponent(version)}`,
        ),

    healthz: () => jsonFetch<{ ok: boolean; envs: string[] }>("/api/healthz"),
}
