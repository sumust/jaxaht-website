// Frontend env registry. One entry per env.
// Pages (Play, Replay) pick the right Board + Controls via this map,
// keeping page code env-agnostic.

import { HanabiBoard } from "./hanabi/Board"
import type { HanabiAction, HanabiState } from "./hanabi/types"
import { LBFBoard } from "./lbf/Board"
import type { LBFAction, LBFState } from "./lbf/types"
import { OvercookedBoard } from "./overcooked/Board"
import type { OvercookedAction, OvercookedState } from "./overcooked/types"

export type EnvComponents<S, A> = {
    Board: React.ComponentType<{ state: S; onAction: (a: A) => void; disabled?: boolean }>
    parseState: (raw: unknown) => S
    defaultSeed?: number
}

export const ENV_REGISTRY: Record<string, EnvComponents<any, any>> = {
    hanabi: {
        Board: HanabiBoard,
        parseState: (raw) => raw as HanabiState,
    },
    "mini-hanabi": {
        Board: HanabiBoard,
        parseState: (raw) => raw as HanabiState,
    },
    lbf: {
        Board: LBFBoard,
        parseState: (raw) => raw as LBFState,
    },
    "overcooked-v1": {
        Board: OvercookedBoard,
        parseState: (raw) => raw as OvercookedState,
    },
}

export function getEnvComponents(envName: string) {
    const entry = ENV_REGISTRY[envName]
    if (!entry) {
        throw new Error(`no frontend components registered for env '${envName}'`)
    }
    return entry
}

export type { HanabiAction, HanabiState, LBFAction, LBFState, OvercookedAction, OvercookedState }
