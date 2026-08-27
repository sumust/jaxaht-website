import argparse
import os
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from agents.bc.bc_lstm import BCLSTMConfig, BCLSTMNetwork, compute_bc_loss, BCLSTMAgent

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_data(data_dir: str):
    data_path = Path(data_dir)
    keys = ["obs", "actions", "avail", "mask"]

    from safetensors.numpy import load_file

    required = [f"{s}_{k}.safetensors" for s in ["train", "val"] for k in keys]
    missing = [f for f in required if not (data_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}")

    arrays = {}
    for split in ["train", "val"]:
        for key in keys:
            raw = load_file(str(data_path / f"{split}_{key}.safetensors"))
            arrays[f"{split}_{key}"] = jnp.array(raw["data"])
    return tuple(arrays[f"{s}_{k}"] for s in ["train", "val"] for k in keys)


def build_color_augmentor(num_colors=5, num_ranks=5, hand_size=5, num_agents=2):
    """Build a JIT-compiled color augmentation function (call once per run).

    Returns a function augment_batch(obs, actions, avail, perms) that
    applies per-episode color permutations to a batch of episodes.

    Hanabi-specific: imports color permutation helpers lazily so non-Hanabi
    BC training (LBF, Overcooked) does not require agents.hanabi.other_play.
    """
    from envs.hanabi.other_play import (
        permute_observation, inverse_permutation,
    )

    hint_start = 2 * hand_size
    discard_per_color = sum({5: [3, 2, 2, 2, 1], 3: [2, 2, 1]}[num_ranks])

    _permute_obs_step = jax.vmap(
        lambda o, p: permute_observation(
            o, p, num_colors=num_colors, num_ranks=num_ranks,
            hand_size=hand_size, num_agents=num_agents,
            discard_entries_per_color=discard_per_color),
        in_axes=(0, None))
    _permute_obs_batch = jax.vmap(_permute_obs_step, in_axes=(0, 0))

    @jax.jit
    def augment_batch(obs_batch, act_batch, avail_batch, perms):
        """Augment a batch of episodes with per-episode color permutations."""
        aug_obs = _permute_obs_batch(obs_batch, perms)

        inv_perms = jax.vmap(inverse_permutation)(perms)
        is_hint = (act_batch >= hint_start) & (act_batch < hint_start + num_colors)
        color_idx = act_batch - hint_start
        safe_idx = jnp.clip(color_idx, 0, num_colors - 1)
        new_color = jax.vmap(lambda ip, idx: ip[idx])(inv_perms, safe_idx)
        aug_act = jnp.where(is_hint, hint_start + new_color, act_batch)

        color_avail = avail_batch[:, :, hint_start:hint_start + num_colors]
        new_color_avail = jax.vmap(
            lambda ca, p: ca[:, p],
            in_axes=(0, 0))(color_avail, perms)
        aug_avail = avail_batch.at[:, :, hint_start:hint_start + num_colors].set(
            new_color_avail)

        return aug_obs, aug_act, aug_avail

    return augment_batch


def augment_epoch(obs, actions, avail, mask, augment_fn, rng, num_aug,
                  num_colors=5):
    """Generate fresh color-augmented training data for one epoch.

    Returns (num_aug+1)*N episodes: original + num_aug permuted copies
    with fresh random permutations each call.
    """
    all_obs = [obs]
    all_act = [actions]
    all_avail = [avail]
    all_mask = [mask]
    n_eps = obs.shape[0]

    for _ in range(num_aug):
        rng, sub_rng = jax.random.split(rng)
        keys = jax.random.split(sub_rng, n_eps)
        perms = jax.vmap(lambda k: jax.random.permutation(k, num_colors))(keys)
        a_obs, a_act, a_avail = augment_fn(obs, actions, avail, perms)
        all_obs.append(a_obs)
        all_act.append(a_act)
        all_avail.append(a_avail)
        all_mask.append(mask)

    return (jnp.concatenate(all_obs), jnp.concatenate(all_act),
            jnp.concatenate(all_avail), jnp.concatenate(all_mask))


def selfplay_eval(network, params, config, num_games=256):
    """Run greedy self-play Hanabi games and return (mean_score, std_score).

    Both agents use the same BC-LSTM weights with greedy action selection.
    Tracks cumulative reward (not terminal state.score) to handle bomb-outs
    correctly, since JaxMARL auto-resets on done.
    """
    from jaxmarl import make
    env = make("hanabi", num_agents=2)

    scores = []
    for game_i in range(num_games):
        rng = jax.random.PRNGKey(game_i * 7919)
        obs_t, state = env.reset(rng)

        carries = {
            "agent_0": (jnp.zeros((config.lstm_dim,)),
                        jnp.zeros((config.lstm_dim,))),
            "agent_1": (jnp.zeros((config.lstm_dim,)),
                        jnp.zeros((config.lstm_dim,))),
        }

        cum_reward = 0.0

        for t in range(200):
            cur_player = int(jnp.argmax(state.cur_player_idx))
            agent_key = f"agent_{cur_player}"
            other_key = f"agent_{1 - cur_player}"

            obs = obs_t[agent_key]
            legal = env.get_legal_moves(state)[agent_key]

            carry = carries[agent_key]
            carry_new, logits = network.apply(
                {'params': params}, carry, obs)
            logits = jnp.where(legal > 0, logits, -1e9)
            action = int(jnp.argmax(logits))
            carries[agent_key] = carry_new

            other_obs = obs_t[other_key]
            other_carry = carries[other_key]
            other_carry_new, _ = network.apply(
                {'params': params}, other_carry, other_obs)
            carries[other_key] = other_carry_new

            actions = {
                "agent_0": jnp.array(action if cur_player == 0 else 0),
                "agent_1": jnp.array(action if cur_player == 1 else 0),
            }
            rng, step_rng = jax.random.split(rng)
            obs_t, state, rewards, dones, infos = env.step(
                step_rng, state, actions)

            cum_reward += float(rewards["agent_0"])

            if bool(dones["__all__"]):
                break

        scores.append(cum_reward)

    return float(np.mean(scores)), float(np.std(scores))


def train(args):
    rng = jax.random.PRNGKey(args.seed)

    defaults = {}
    if args.config:
        defaults = load_config(args.config)
        print(f"[train_bc] loaded config from {args.config}")

    (train_obs, train_actions, train_avail, train_mask,
     val_obs, val_actions, val_avail, val_mask) = load_data(args.data_dir)

    obs_dim = int(train_obs.shape[-1])
    action_dim = int(train_avail.shape[-1])

    # Color augmentation setup (Hanabi-specific, per-epoch fresh permutations)
    n_aug = getattr(args, 'color_aug', 0) or 0
    augment_fn = None
    if n_aug > 0:
        augment_fn = build_color_augmentor()
        print(f"[train_bc] color augmentation: {n_aug} permutations/epoch")

    base_obs = train_obs
    base_actions = train_actions
    base_avail = train_avail
    base_mask = train_mask

    print(f"[train_bc] obs_dim={obs_dim} action_dim={action_dim}")
    print(f"[train_bc] train: {train_obs.shape[0]} episodes, "
          f"val: {val_obs.shape[0]} episodes")

    dropout = getattr(args, 'dropout', 0.0) or defaults.get("dropout", 0.0)

    # Resolve pre/post dims: CLI list > YAML list > YAML scalar > default
    def _resolve_dims(cli_val, yaml_key_multi, yaml_key_single, default_single):
        if cli_val is not None:
            return tuple(cli_val)
        multi = defaults.get(yaml_key_multi)
        if multi:
            return tuple(multi)
        single = defaults.get(yaml_key_single, default_single)
        return (single,)

    pre_dims = _resolve_dims(args.preprocess_dim, "preprocess_dims",
                             "preprocess_dim", 256)
    post_dims = _resolve_dims(args.postprocess_dim, "postprocess_dims",
                              "postprocess_dim", 64)
    config = BCLSTMConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lstm_dim=args.lstm_dim or defaults.get("lstm_dim", 128),
        dropout_rate=dropout,
        preprocess_dims=pre_dims,
        postprocess_dims=post_dims,
    )
    print(f"[train_bc] config: pre={pre_dims} lstm={config.lstm_dim} "
          f"post={post_dims} dropout={dropout}")

    network = BCLSTMNetwork(
        action_dim=config.action_dim,
        preprocess_dims=config.effective_preprocess_dims,
        lstm_dim=config.lstm_dim,
        postprocess_dims=config.effective_postprocess_dims,
        dropout_rate=config.dropout_rate,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_carry = (jnp.zeros((config.lstm_dim,)), jnp.zeros((config.lstm_dim,)))
    dummy_obs = jnp.zeros((config.obs_dim,))
    variables = network.init(init_rng, dummy_carry, dummy_obs)

    lr_schedule_type = getattr(args, 'lr_schedule', 'flat')
    optimizer_type = getattr(args, 'optimizer', 'adam')
    batch_size = min(args.batch_size, base_obs.shape[0])
    if lr_schedule_type == "linear":
        eps_per_epoch = base_obs.shape[0] * (1 + n_aug)
        n_updates = args.epochs * max(1, eps_per_epoch // batch_size)
        schedule = optax.linear_schedule(
            init_value=args.lr, end_value=args.lr * 0.02,
            transition_steps=int(n_updates * 0.9))
    else:
        schedule = args.lr
    if optimizer_type == "adamw":
        tx = optax.adamw(schedule)
    else:
        tx = optax.adam(schedule)
    state = TrainState.create(
        apply_fn=network.apply, params=variables['params'], tx=tx)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"[train_bc] {param_count:,} parameters")

    use_legal_mask = getattr(args, 'legal_mask_loss', False)
    selfplay_games = getattr(args, 'selfplay_eval_games', 0) or 0
    selfplay_every = getattr(args, 'selfplay_every', 1) or 1
    print(f"[train_bc] optimizer={optimizer_type} lr_schedule={lr_schedule_type} "
          f"legal_mask_loss={use_legal_mask}")
    if selfplay_games > 0:
        print(f"[train_bc] self-play eval: {selfplay_games} games "
              f"every {selfplay_every} epoch(s)")

    best_nn_acc = 0.0     # non-noop accuracy for model selection
    best_nn_params = None
    best_sp_score = -1.0  # self-play score for separate checkpoint
    best_sp_params = None

    @jax.jit
    def train_step(state, batch_obs, batch_actions, batch_avail, batch_mask):
        init_carry = (
            jnp.zeros((batch_obs.shape[0], config.lstm_dim)),
            jnp.zeros((batch_obs.shape[0], config.lstm_dim)),
        )
        def loss_fn(params):
            return compute_bc_loss(
                params, network, init_carry,
                batch_obs, batch_actions, batch_avail, batch_mask,
                legal_mask_loss=use_legal_mask,
            )
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_metrics(params, obs, actions, avail, mask):
        """Return (total_accuracy, non_noop_accuracy)."""
        init_carry = (
            jnp.zeros((obs.shape[0], config.lstm_dim)),
            jnp.zeros((obs.shape[0], config.lstm_dim)),
        )
        _, seq_len, _ = obs.shape
        def step_fn(carry, t):
            carry, logits = network.apply(
                {'params': params}, carry, obs[:, t, :])
            if use_legal_mask:
                logits = logits - (1.0 - avail[:, t, :]) * 1e10
            preds = jnp.argmax(logits, axis=-1)
            correct = (preds == actions[:, t]) & mask[:, t]
            valid = mask[:, t]
            return carry, (correct, valid)
        _, (all_correct, all_valid) = jax.lax.scan(
            step_fn, init_carry, jnp.arange(seq_len))
        # all_correct, all_valid: (seq_len, batch_size)
        total_acc = all_correct.sum() / jnp.maximum(all_valid.sum(), 1.0)

        # Non-noop: exclude timesteps where ground-truth action is noop
        noop_id = action_dim - 1
        non_noop = (actions.T != noop_id)  # (seq_len, batch)
        nn_correct = (all_correct * non_noop).sum()
        nn_valid = (all_valid * non_noop).sum()
        non_noop_acc = nn_correct / jnp.maximum(nn_valid, 1.0)

        return total_acc, non_noop_acc

    for epoch in range(args.epochs):
        t0 = time.time()

        if n_aug > 0:
            rng, aug_rng = jax.random.split(rng)
            epoch_obs, epoch_act, epoch_avail, epoch_mask = augment_epoch(
                base_obs, base_actions, base_avail, base_mask,
                augment_fn, aug_rng, n_aug)
        else:
            epoch_obs = base_obs
            epoch_act = base_actions
            epoch_avail = base_avail
            epoch_mask = base_mask

        rng, perm_rng = jax.random.split(rng)
        n_epoch = epoch_obs.shape[0]
        perm = jax.random.permutation(perm_rng, n_epoch)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_epoch, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue
            state, loss = train_step(
                state, epoch_obs[idx], epoch_act[idx],
                epoch_avail[idx], epoch_mask[idx],
            )
            epoch_loss += float(loss)
            n_batches += 1

        total_acc, nn_acc = [float(x) for x in eval_metrics(
            state.params, val_obs, val_actions, val_avail, val_mask)]
        avg_loss = epoch_loss / max(n_batches, 1)
        dt = time.time() - t0

        sp_str = ""
        if selfplay_games > 0 and (epoch + 1) % selfplay_every == 0:
            sp_t0 = time.time()
            sp_mean, sp_std = selfplay_eval(
                network, state.params, config, num_games=selfplay_games)
            sp_dt = time.time() - sp_t0
            sp_str = f" sp={sp_mean:.2f}+/-{sp_std:.2f} ({sp_dt:.1f}s)"
            if sp_mean > best_sp_score:
                best_sp_score = sp_mean
                best_sp_params = jax.tree.map(lambda x: x.copy(), state.params)

        print(f"  epoch {epoch+1:3d}/{args.epochs}: loss={avg_loss:.4f} "
              f"acc={total_acc:.3f} nn_acc={nn_acc:.3f}{sp_str} ({dt:.1f}s)")

        if nn_acc > best_nn_acc:
            best_nn_acc = nn_acc
            best_nn_params = jax.tree.map(lambda x: x.copy(), state.params)

    print(f"\n[train_bc] best non-noop val accuracy: {best_nn_acc:.4f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    config_path = args.output.replace('.safetensors', '.yaml')
    with open(config_path, 'w') as f:
        yaml.dump({
            'obs_dim': config.obs_dim,
            'action_dim': config.action_dim,
            'preprocess_dims': list(config.effective_preprocess_dims),
            'lstm_dim': config.lstm_dim,
            'postprocess_dims': list(config.effective_postprocess_dims),
            'val_accuracy': float(best_nn_acc),
            'data_dir': args.data_dir,
            'epochs': args.epochs,
            'lr': args.lr,
            'seed': args.seed,
        }, f)
    print(f"[train_bc] saved config to {config_path}")

    agent = BCLSTMAgent(config, params=best_nn_params)
    agent.save_weights(args.output)
    print(f"[train_bc] saved best-val weights to {args.output}")

    if best_sp_params is not None:
        sp_path = args.output.replace('.safetensors', '_selfplay.safetensors')
        sp_config_path = sp_path.replace('.safetensors', '.yaml')
        with open(sp_config_path, 'w') as f:
            yaml.dump({
                'obs_dim': config.obs_dim,
                'action_dim': config.action_dim,
                'preprocess_dims': list(config.effective_preprocess_dims),
                'lstm_dim': config.lstm_dim,
                'postprocess_dims': list(config.effective_postprocess_dims),
                'selfplay_score': float(best_sp_score),
                'data_dir': args.data_dir,
                'epochs': args.epochs,
                'lr': args.lr,
                'seed': args.seed,
            }, f)
        sp_agent = BCLSTMAgent(config, params=best_sp_params)
        sp_agent.save_weights(sp_path)
        print(f"[train_bc] best self-play score: {best_sp_score:.2f}")
        print(f"[train_bc] saved best-selfplay weights to {sp_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train BC-LSTM from preprocessed safetensors data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_dir', required=True,
                        help="Directory with {train,val}_{obs,actions,avail,mask}.safetensors")
    parser.add_argument('--output', required=True,
                        help="Output path for .safetensors weights")
    parser.add_argument('--config', type=str, default=None,
                        help="YAML config with architecture defaults "
                             "(agents/bc/configs/*.yaml)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preprocess_dim', type=int, nargs='+', default=None,
                        help="Pre-LSTM dense dims (e.g. 512 512 for two layers)")
    parser.add_argument('--lstm_dim', type=int, default=None)
    parser.add_argument('--postprocess_dim', type=int, nargs='+', default=None,
                        help="Post-LSTM dense dims (e.g. 256 256 for two layers)")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout rate (0.0 = no dropout)")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw'],
                        help="Optimizer (default: adam)")
    parser.add_argument('--lr_schedule', type=str, default='flat',
                        choices=['flat', 'linear'],
                        help="LR schedule (default: flat)")
    parser.add_argument('--legal_mask_loss', action='store_true', default=False,
                        help="Mask illegal actions in loss (default: off)")
    parser.add_argument('--color_aug', type=int, default=0,
                        help="Number of color-permutation augmentations per epoch "
                             "(0=off, 4=standard for Hanabi 5-color)")
    parser.add_argument('--selfplay_eval_games', type=int, default=0,
                        help="Self-play games per eval (0=off, 256=standard)")
    parser.add_argument('--selfplay_every', type=int, default=1,
                        help="Run self-play eval every N epochs (default: 1)")
    train(parser.parse_args())
