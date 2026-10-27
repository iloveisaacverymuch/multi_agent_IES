import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_maddpg_baseline import (  # noqa: E402
    ACT_DIMS,
    OBS_DIMS,
    env_action,
    flatten_states,
    make_env,
    make_weights,
    scalarized_rewards,
    update_policy_archive,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTION_EPSILON = 1e-6


def initialize_linear(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class IPPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            initialize_linear(nn.Linear(obs_dim, hidden_dim), np.sqrt(2.0)),
            nn.Tanh(),
            initialize_linear(nn.Linear(hidden_dim, hidden_dim), np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.mean = initialize_linear(nn.Linear(hidden_dim, act_dim), 0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def distribution(self, obs):
        mean = self.mean(self.backbone(obs))
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return Normal(mean, log_std.exp())

    @staticmethod
    def squash_log_prob(distribution, raw_action, action):
        correction = torch.log(1.0 - action.pow(2) + ACTION_EPSILON)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def forward(self, obs, evaluation=False):
        distribution = self.distribution(obs)
        if evaluation:
            return torch.tanh(distribution.mean)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self.squash_log_prob(distribution, raw_action, action)
        return action, log_prob

    def evaluate_actions(self, obs, action):
        bounded_action = action.clamp(-1.0 + ACTION_EPSILON, 1.0 - ACTION_EPSILON)
        raw_action = 0.5 * (
            torch.log1p(bounded_action) - torch.log1p(-bounded_action)
        )
        distribution = self.distribution(obs)
        log_prob = self.squash_log_prob(distribution, raw_action, bounded_action)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy


class IPPOValue(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            initialize_linear(nn.Linear(obs_dim, hidden_dim), np.sqrt(2.0)),
            nn.Tanh(),
            initialize_linear(nn.Linear(hidden_dim, hidden_dim), np.sqrt(2.0)),
            nn.Tanh(),
            initialize_linear(nn.Linear(hidden_dim, 1), 1.0),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class IPPO:
    """Independent PPO: each agent uses only its local observation and return."""

    def __init__(self, args):
        self.device = torch.device(args.device)
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_ratio = args.clip_ratio
        self.ppo_epochs = args.ppo_epochs
        self.minibatch_size = args.minibatch_size
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm

        self.actors = nn.ModuleList(
            [IPPOActor(OBS_DIMS[i], ACT_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.values = nn.ModuleList(
            [IPPOValue(OBS_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=args.actor_lr) for actor in self.actors
        ]
        self.value_opts = [
            torch.optim.Adam(value.parameters(), lr=args.critic_lr) for value in self.values
        ]

    def act(self, obs, evaluation=False):
        actions = []
        log_probs = []
        values = []
        with torch.no_grad():
            for agent_id in range(3):
                obs_tensor = torch.as_tensor(
                    obs[agent_id], dtype=torch.float32, device=self.device
                )
                if evaluation:
                    action = self.actors[agent_id](obs_tensor, evaluation=True)
                else:
                    action, log_prob = self.actors[agent_id](obs_tensor)
                    log_probs.append(float(log_prob.item()))
                    values.append(float(self.values[agent_id](obs_tensor).item()))
                actions.append(action.cpu().numpy().astype(np.float32))
        if evaluation:
            return actions
        return actions, log_probs, values

    def _advantages(self, agent_id, rollout, next_obs, terminal):
        rewards = np.asarray(rollout[agent_id]["rewards"], dtype=np.float32)
        values = np.asarray(rollout[agent_id]["values"], dtype=np.float32)
        dones = np.asarray(rollout[agent_id]["dones"], dtype=np.float32)
        with torch.no_grad():
            if terminal:
                last_value = 0.0
            else:
                next_tensor = torch.as_tensor(
                    next_obs[agent_id], dtype=torch.float32, device=self.device
                )
                last_value = float(self.values[agent_id](next_tensor).item())

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * nonterminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[step] = gae
        return advantages, advantages + values

    def update(self, rollout, next_obs, terminal):
        for agent_id in range(3):
            advantages, returns = self._advantages(agent_id, rollout, next_obs, terminal)
            obs = torch.as_tensor(
                np.asarray(rollout[agent_id]["obs"], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            actions = torch.as_tensor(
                np.asarray(rollout[agent_id]["actions"], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            old_log_probs = torch.as_tensor(
                rollout[agent_id]["log_probs"], dtype=torch.float32, device=self.device
            )
            advantages = torch.as_tensor(
                advantages, dtype=torch.float32, device=self.device
            )
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            sample_count = len(obs)
            for _ in range(self.ppo_epochs):
                indices = np.random.permutation(sample_count)
                for start in range(0, sample_count, self.minibatch_size):
                    batch_indices = torch.as_tensor(
                        indices[start : start + self.minibatch_size],
                        dtype=torch.long,
                        device=self.device,
                    )
                    new_log_probs, entropy = self.actors[agent_id].evaluate_actions(
                        obs[batch_indices], actions[batch_indices]
                    )
                    ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                    unclipped = ratio * advantages[batch_indices]
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * advantages[batch_indices]
                    actor_loss = -torch.minimum(unclipped, clipped).mean()
                    actor_loss -= self.entropy_coef * entropy.mean()
                    self.actor_opts[agent_id].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[agent_id].parameters(), self.max_grad_norm
                    )
                    self.actor_opts[agent_id].step()

                    predicted_values = self.values[agent_id](obs[batch_indices])
                    value_loss = self.value_coef * F.mse_loss(
                        predicted_values, returns[batch_indices]
                    )
                    self.value_opts[agent_id].zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.values[agent_id].parameters(), self.max_grad_norm
                    )
                    self.value_opts[agent_id].step()


def empty_rollout():
    return [
        {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        for _ in range(3)
    ]


def horizon_reached(step, max_episode_steps):
    return max_episode_steps > 0 and step >= max_episode_steps


def evaluate(agent, args):
    environment = make_env(args)
    state_res, state_com, state_ind = environment.reset_train()
    done = False
    step = 0
    total = np.zeros(2, dtype=np.float64)
    while not done and not horizon_reached(step, args.max_episode_steps):
        _, obs = flatten_states(state_res, state_com, state_ind)
        actions = agent.act(obs, evaluation=True)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
        )
        total += np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
        state_res, state_com, state_ind = next_res, next_com, next_ind
        step += 1
    return total


def train_one_weight(weight, weight_index, args, seed, policy_archive):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    agent = IPPO(args)
    episode_rows = []
    final_eval = None

    for episode in range(1, args.episodes + 1):
        environment = make_env(args)
        state_res, state_com, state_ind = environment.reset_train()
        done = False
        step = 0
        episode_objectives = np.zeros(2, dtype=np.float64)
        rollout = empty_rollout()
        next_obs = None

        while not done and not horizon_reached(step, args.max_episode_steps):
            _, obs = flatten_states(state_res, state_com, state_ind)
            actions, log_probs, values = agent.act(obs)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
                env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
            )
            _, next_obs = flatten_states(next_res, next_com, next_ind)
            rewards = scalarized_rewards(obj_r, obj_c, obj_i, weight, args.reward_scale)
            for agent_id in range(3):
                rollout[agent_id]["obs"].append(obs[agent_id])
                rollout[agent_id]["actions"].append(actions[agent_id])
                rollout[agent_id]["log_probs"].append(log_probs[agent_id])
                rollout[agent_id]["rewards"].append(rewards[agent_id])
                rollout[agent_id]["dones"].append(float(done))
                rollout[agent_id]["values"].append(values[agent_id])
            episode_objectives += np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1

        if next_obs is None:
            raise RuntimeError("IPPO episode produced no transitions")
        agent.update(rollout, next_obs, terminal=done)

        should_evaluate = args.archive_every_episode or episode % args.eval_every == 0
        eval_objectives = evaluate(agent, args) if should_evaluate else np.array([np.nan, np.nan])
        if should_evaluate:
            final_eval = eval_objectives
            if args.archive_every_episode:
                update_policy_archive(
                    policy_archive,
                    eval_objectives,
                    agent,
                    {
                        "weight_index": weight_index,
                        "episode": episode,
                        "weight_cost": float(weight[0]),
                        "weight_carbon": float(weight[1]),
                    },
                )
        episode_rows.append(
            {
                "episode": episode,
                "train_cost": episode_objectives[0],
                "train_carbon": episode_objectives[1],
                "eval_cost": eval_objectives[0],
                "eval_carbon": eval_objectives[1],
            }
        )

    if final_eval is None or not args.archive_every_episode:
        final_eval = evaluate(agent, args)
    return agent, final_eval, episode_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_archive(policy_archive, args, final_dir):
    policy_archive.sort(key=lambda entry: entry["objectives"][0])
    objectives = np.asarray([entry["objectives"] for entry in policy_archive])
    metadata = []
    for pareto_index, entry in enumerate(policy_archive):
        policy_file = f"EP_policy_{pareto_index}.pt"
        torch.save(
            {
                "format": "ippo_state_dict_v1",
                "hidden_dim": args.hidden_dim,
                "actor_state_dicts": [
                    entry["actors"][agent_id].state_dict() for agent_id in range(3)
                ],
            },
            final_dir / policy_file,
        )
        metadata.append(
            {
                "pareto_index": pareto_index,
                "policy_file": policy_file,
                "cost": entry["objectives"][0],
                "carbon": entry["objectives"][1],
                "weight_cost": entry["weight_cost"],
                "weight_carbon": entry["weight_carbon"],
                "weight_index": entry["weight_index"],
                "episode": entry["episode"],
            }
        )
    np.savetxt(final_dir / "objs.txt", objectives, delimiter=",", fmt="%.6f")
    write_csv(final_dir / "metadata.csv", metadata)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a fixed-weight IPPO baseline.")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "baseline_ippo")
    parser.add_argument("--num-weights", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=162)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--archive-every-episode", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument(
        "--market-mechanism",
        choices=["standard", "carbon_dividend_v3"],
        default="carbon_dividend_v3",
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.ppo_epochs < 1 or args.minibatch_size < 1:
        raise ValueError("PPO epochs and minibatch size must be positive")
    save_dir = args.save_dir if args.save_dir.is_absolute() else PROJECT_ROOT / args.save_dir
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["save_dir"] = str(save_dir)
    config["algorithm"] = "fixed_weight_ippo"
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    started = time.time()
    weights = make_weights(args.num_weights)
    policy_archive = []
    all_evaluations = []
    for weight_index, weight in enumerate(weights):
        agent, objectives, rows = train_one_weight(
            weight,
            weight_index,
            args,
            args.seed + weight_index * 1000,
            policy_archive,
        )
        if not args.archive_every_episode:
            update_policy_archive(
                policy_archive,
                objectives,
                agent,
                {
                    "weight_index": weight_index,
                    "episode": args.episodes,
                    "weight_cost": float(weight[0]),
                    "weight_carbon": float(weight[1]),
                },
            )
        all_evaluations.extend(
            [
                [row["eval_cost"], row["eval_carbon"]]
                for row in rows
                if np.isfinite(row["eval_cost"]) and np.isfinite(row["eval_carbon"])
            ]
        )
        write_csv(save_dir / f"weight_{weight_index}_episodes.csv", rows)
        print(
            f"weight {weight_index + 1}/{len(weights)}: "
            f"w={weight.tolist()}, objectives={objectives.tolist()}, archive={len(policy_archive)}",
            flush=True,
        )

    save_archive(policy_archive, args, final_dir)
    evaluations = np.asarray(all_evaluations, dtype=np.float64).reshape(-1, 2)
    np.savetxt(save_dir / "all_objs.txt", evaluations, delimiter=",", fmt="%.6f")
    elapsed = time.time() - started
    (save_dir / "launcher_elapsed_seconds.txt").write_text(
        f"{elapsed:.6f}\n", encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "algorithm": "fixed_weight_ippo",
        "seed": args.seed,
        "archive_points": len(policy_archive),
        "training_episodes": args.num_weights * args.episodes,
        "elapsed_seconds": elapsed,
    }
    (save_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
