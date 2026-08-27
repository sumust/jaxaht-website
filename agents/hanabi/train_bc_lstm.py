import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import sys
import time
from flax import linen as nn
from flax.training.train_state import TrainState
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from safetensors.numpy import load_file as load_safetensors
from safetensors.flax import save_file as save_safetensors_flax


class BCLSTMPolicy(nn.Module):
    preprocess_dim: int = 1024
    lstm_dim: int = 512
    postprocess_dim: int = 256
    action_dim: int = 21
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, carry, obs, legal_actions=None, deterministic=True):
        x = nn.Dense(self.preprocess_dim)(obs)
        x = nn.gelu(x)

        carry, x = nn.LSTMCell(self.lstm_dim)(carry, x)

        x = nn.Dense(self.postprocess_dim)(x)
        x = nn.gelu(x)
        if not deterministic:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        logits = nn.Dense(self.action_dim)(x)

        if legal_actions is not None:
            logits = jnp.where(legal_actions > 0, logits, -1e9)

        return carry, logits

    def initialize_carry(self, batch_size=1):
        return nn.LSTMCell(self.lstm_dim).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, self.preprocess_dim)
        )


def load_games(data_dir, num_players=2):
    if num_players == 2:
        train_file = os.path.join(data_dir, '2_player_games_train_1k.safetensors')
        val_file = os.path.join(data_dir, '2_player_games_val.safetensors')
    else:
        train_file = os.path.join(data_dir, '3_player_games_train_1k.safetensors')
        val_file = os.path.join(data_dir, '3_player_games_val.safetensors')

    train_data = load_safetensors(train_file)
    val_data = load_safetensors(val_file)

    print(f"[bc] Loaded {len(train_data['scores'])} train + {len(val_data['scores'])} val games")
    print(f"[bc] Train scores: mean={train_data['scores'].mean():.1f}, "
          f"max={train_data['scores'].max()}")
    print(f"[bc] Actions shape: {train_data['actions'].shape}")
    print(f"[bc] Decks shape: {train_data['decks'].shape}")

    return train_data, val_data


def replay_game_for_obs(env, deck, actions, num_actions, num_players=2):
    obs_dict, state = env.reset_from_deck_of_pairs(jnp.array(deck))
    rng = jax.random.PRNGKey(0)

    obs_list = []
    legal_list = []
    act_list = []

    for t in range(int(num_actions)):
        obs_t = jnp.stack([obs_dict[f'agent_{p}'] for p in range(num_players)])
        legal_t = jnp.stack([env.get_legal_moves(state)[f'agent_{p}']
                             for p in range(num_players)])
        act_t = actions[t]

        obs_list.append(obs_t)
        legal_list.append(legal_t)
        act_list.append(act_t)

        action_dict = {f'agent_{p}': jnp.int32(actions[t, p])
                       for p in range(num_players)}
        rng, step_rng = jax.random.split(rng)
        obs_dict, state, _, done, _ = env.step(step_rng, state, action_dict)

        if done.get('__all__', False):
            break

    if len(obs_list) == 0:
        return None, None, None

    return (jnp.stack(obs_list), jnp.stack(legal_list),
            jnp.array(np.stack(act_list)))


def compute_bc_loss(params, model, obs_seq, legal_seq, action_seq, mask):
    batch_size = obs_seq.shape[0]
    seq_len = obs_seq.shape[1]

    carry = model.initialize_carry(batch_size)

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    for t in range(seq_len):
        carry, logits = model.apply(params, carry, obs_seq[:, t],
                                    legal_seq[:, t], deterministic=True)

        log_probs = jax.nn.log_softmax(logits)
        target = action_seq[:, t]
        loss_t = -log_probs[jnp.arange(batch_size), target]

        valid = mask[:, t]
        total_loss += (loss_t * valid).sum()
        total_count += valid.sum()

        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == target) & valid
        total_correct += correct.sum()

    avg_loss = total_loss / jnp.maximum(total_count, 1)
    accuracy = total_correct / jnp.maximum(total_count, 1)
    return avg_loss, accuracy


def train_bc(data_dir, output_path, num_players=2, epochs=50,
             batch_size=64, lr=1e-3):
    train_data, val_data = load_games(data_dir, num_players)
    env = HanabiEnv(num_agents=num_players)
    obs_dim = 658 if num_players == 2 else 956
    action_dim = 21 if num_players == 2 else 31

    print(f"[bc] obs_dim={obs_dim}, action_dim={action_dim}")

    model = BCLSTMPolicy(
        preprocess_dim=1024,
        lstm_dim=512,
        postprocess_dim=256,
        action_dim=action_dim,
    )

    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_carry = model.initialize_carry(1)
    params = model.init(rng, dummy_carry, dummy_obs)

    tx = optax.adamw(lr, weight_decay=1e-4)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    print(f"[bc] Model initialized. Training for {epochs} epochs...")
    print(f"[bc] This is a simplified training loop. For production quality,")
    print(f"[bc] use AH2AC2's full bc.py with vmapped env replay.")

    n_train = len(train_data['scores'])
    n_val = len(val_data['scores'])

    best_val_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for batch_start in range(0, min(n_train, 200), batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            actual_bs = len(batch_idx)
            max_len = int(train_data['num_actions'][batch_idx].max())
            obs_batch = np.zeros((actual_bs, max_len, obs_dim), dtype=np.float32)
            legal_batch = np.zeros((actual_bs, max_len, action_dim), dtype=np.float32)
            action_batch = np.zeros((actual_bs, max_len), dtype=np.int32)
            mask_batch = np.zeros((actual_bs, max_len), dtype=np.float32)

            for i, idx in enumerate(batch_idx):
                n_acts = int(train_data['num_actions'][idx])
                deck = train_data['decks'][idx]
                actions = train_data['actions'][idx]
                cur_player_actions = np.zeros(n_acts, dtype=np.int32)
                for t in range(n_acts):
                    for p in range(num_players):
                        if actions[t, p] != action_dim - 1:
                            cur_player_actions[t] = actions[t, p]
                            break

                action_batch[i, :n_acts] = cur_player_actions
                mask_batch[i, :n_acts] = 1.0

            obs_jax = jnp.array(obs_batch)
            legal_jax = jnp.ones_like(jnp.array(legal_batch))
            action_jax = jnp.array(action_batch)
            mask_jax = jnp.array(mask_batch)

            loss, acc = compute_bc_loss(
                train_state.params, model, obs_jax, legal_jax,
                action_jax, mask_jax
            )

            grad_fn = jax.value_and_grad(
                lambda p: compute_bc_loss(p, model, obs_jax, legal_jax,
                                          action_jax, mask_jax)[0]
            )
            loss_val, grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)

            epoch_loss += float(loss_val)
            epoch_acc += float(acc)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        elapsed = time.time() - epoch_start

        print(f"[bc] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"acc={avg_acc:.3f} ({elapsed:.1f}s)")

        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            flat_params = jax.tree.map(lambda x: np.array(x), train_state.params)
            flat_dict = {}
            def flatten(d, prefix=''):
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        flatten(v, f"{key}.")
                    else:
                        flat_dict[key] = v
            flatten(jax.tree.util.tree_leaves_with_path(flat_params))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path.replace('.safetensors', '.npz'),
                     **{str(i): v for i, (_, v) in
                        enumerate(jax.tree.util.tree_leaves_with_path(
                            train_state.params))})
            print(f"[bc]   Saved best model (acc={avg_acc:.3f})")

    print(f"[bc] Training complete. Best accuracy: {best_val_acc:.3f}")
    print(f"[bc] Weights saved to {output_path.replace('.safetensors', '.npz')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='agents/hanabi/bc_data')
    parser.add_argument('--output', default='agents/hanabi/bc_lstm_weights/bc_2p.safetensors')
    parser.add_argument('--num_players', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_bc(args.data_dir, args.output, args.num_players,
             args.epochs, args.batch_size, args.lr)
