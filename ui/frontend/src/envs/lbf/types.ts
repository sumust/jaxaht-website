// Mirror of backend/envs/lbf.py::LBFRenderer.serialize_state.

export type LBFAgent = {
    x: number
    y: number
    level: number
}

export type LBFFood = {
    x: number
    y: number
    level: number
    eaten: boolean
}

export type LBFState = {
    grid_size: number
    agents: LBFAgent[]
    food: LBFFood[]
    avail_actions: boolean[]   // [NOOP, UP, DOWN, LEFT, RIGHT, LOAD]
    step_count: number
}

export type LBFAction =
    | { type: "move"; dir: "up" | "down" | "left" | "right" | "load" | "wait" }
