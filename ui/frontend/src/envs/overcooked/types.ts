// Mirror of backend/envs/overcooked.py::OvercookedRenderer.serialize_state.

export type OvercookedAgent = {
    x: number
    y: number
    dir: number     // 0=up, 1=down, 2=right, 3=left in JaxMARL Overcooked
    holding: number // inventory: 0=nothing, 1=onion, 2=plate, 3=soup
}

export type OvercookedPot = {
    x: number
    y: number
}

export type OvercookedState = {
    layout: string
    height: number
    width: number
    agents: OvercookedAgent[]
    wall_map: boolean[][]
    pots: OvercookedPot[]
    step_count: number
    avail_actions: boolean[]
}

export type OvercookedAction =
    | { type: "move"; dir: "up" | "down" | "left" | "right" | "stay" | "interact" }
    | { type: "noop" }
