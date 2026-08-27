// Mirror of the payload returned by HanabiRenderer.serialize_state.
// Keep in sync with backend/envs/hanabi.py.

export type HanabiCard = {
    color: number | null
    rank: number | null
    known: boolean
}

export type HanabiBelief = {
    possible: { color: number; rank: number }[]
    color_hinted: boolean
    rank_hinted: boolean
    color: number | null
    rank: number | null
}

export type HanabiState = {
    variant: "full" | "mini"
    num_colors: number
    num_ranks: number
    hand_size: number
    color_names: string[]
    fireworks: number[]          // length = num_colors. Value = highest-played rank + 1, 0 if nothing
    info_tokens: number
    max_info_tokens: number
    life_tokens: number
    max_life_tokens: number
    current_player: number       // 0 = human, 1 = partner
    human_hand_slots: number
    partner_hand: HanabiCard[]
    human_beliefs: HanabiBelief[]
    discard_pile: HanabiCard[]
    deck_size: number
    num_cards_discarded: number
    score: number
}

export type HanabiAction =
    | { type: "play"; slot: number }
    | { type: "discard"; slot: number }
    | { type: "hint_color"; color: number }
    | { type: "hint_rank"; rank: number }
    | { type: "noop" }

export type HanabiScore = {
    score: number
    max_score: number
    info_tokens: number
    life_tokens: number
    fireworks: number[]
}
