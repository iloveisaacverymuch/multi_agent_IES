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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_maddpg_baseline import (  # noqa: E402
    ACT_DIMS,
    OBS_DIMS,
    Actor,
    Critic,
    ReplayBuffer,
    env_action,
    flatten_states,
    make_env,
    make_weights,
    scalarized_rewards,
    update_policy_archive,
)


class MATD3:
    """Multi-agent TD3 with decentralized actors and twin centralized critics."""

    def __init__(self, args):
        self.device = torch.device(args.device)
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.update_count = 0

        self.actors = nn.ModuleList(
            [Actor(OBS_DIMS[i], ACT_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.target_actors = nn.ModuleList(
            [Actor(OBS_DIMS[i], ACT_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.critics_1 = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)
        self.critics_2 = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)
        self.target_critics_1 = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)
        self.target_critics_2 = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)

        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics_1.load_state_dict(self.critics_1.state_dict())
        self.target_critics_2.load_state_dict(self.critics_2.state_dict())
        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=args.actor_lr) for actor in self.actors
        ]
        self.critic_opts = [
            torch.optim.Adam(
                list(self.critics_1[i].parameters()) + list(self.critics_2[i].parameters()),
                lr=args.critic_lr,
            )
            for i in range(3)
        ]

    def act(self, obs, noise_std=0.0):
        actions = []
        with torch.no_grad():
            for agent_id, actor in enumerate(self.actors):
                obs_tensor = torch.as_tensor(
                    obs[agent_id], dtype=torch.float32, device=self.device
                )
                action = actor(obs_tensor).cpu().numpy()
                if noise_std > 0.0:
                    action += np.random.normal(0.0, noise_std, size=action.shape)
                actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
        return actions

    def update(self, replay, batch_size):
        batch = replay.sample(batch_size, self.device)
        joint_actions = torch.cat(batch["actions"], dim=-1)

        with torch.no_grad():
            next_actions = []
            for agent_id in range(3):
                action = self.target_actors[agent_id](batch["next_obs"][agent_id])
                noise = torch.randn_like(action).mul(self.policy_noise)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_actions.append((action + noise).clamp(-1.0, 1.0))
            next_joint_actions = torch.cat(next_actions, dim=-1)

        for agent_id in range(3):
            with torch.no_grad():
                target_q_1 = self.target_critics_1[agent_id](
                    batch["next_states"], next_joint_actions
                )
                target_q_2 = self.target_critics_2[agent_id](
                    batch["next_states"], next_joint_actions
                )
                target_q = torch.minimum(target_q_1, target_q_2)
                target = batch["rewards"][:, agent_id : agent_id + 1] + self.gamma * (
                    1.0 - batch["dones"]
                ) * target_q

            current_q_1 = self.critics_1[agent_id](batch["states"], joint_actions)
            current_q_2 = self.critics_2[agent_id](batch["states"], joint_actions)
            critic_loss = F.mse_loss(current_q_1, target) + F.mse_loss(current_q_2, target)
            self.critic_opts[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.critics_1[agent_id].parameters())
                + list(self.critics_2[agent_id].parameters()),
                1.0,
            )
            self.critic_opts[agent_id].step()

        self.update_count += 1
        if self.update_count % self.policy_delay != 0:
            return

        for agent_id in range(3):
            actor_actions = []
            for other_id in range(3):
                action = self.actors[other_id](batch["obs"][other_id])
                actor_actions.append(action if other_id == agent_id else action.detach())
            for parameter in self.critics_1[agent_id].parameters():
                parameter.requires_grad_(False)
            actor_loss = -self.critics_1[agent_id](
                batch["states"], torch.cat(actor_actions, dim=-1)
            ).mean()
            self.actor_opts[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1.0)
            self.actor_opts[agent_id].step()
            for parameter in self.critics_1[agent_id].parameters():
                parameter.requires_grad_(True)

        self.soft_update()

    def soft_update(self):
        groups = (
            (self.actors, self.target_actors),
            (self.critics_1, self.target_critics_1),
            (self.critics_2, self.target_critics_2),
        )
        for sources, targets in groups:
            for source, target in zip(sources, targets):
                for source_parameter, target_parameter in zip(
                    source.parameters(), target.parameters()
                ):
                    target_parameter.data.mul_(1.0 - self.tau).add_(
                        self.tau * source_parameter.data
                    )


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
        actions = agent.act(obs, noise_std=0.0)
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
    agent = MATD3(args)
    replay = ReplayBuffer(args.replay_size, sum(OBS_DIMS), OBS_DIMS, ACT_DIMS)
    episode_rows = []
    global_step = 0
    final_eval = None

    for episode in range(1, args.episodes + 1):
        environment = make_env(args)
        state_res, state_com, state_ind = environment.reset_train()
        done = False
        step = 0
        episode_objectives = np.zeros(2, dtype=np.float64)
        noise_std = max(
            args.noise_min,
            args.noise_std * (1.0 - episode / max(args.episodes, 1)),
        )
        while not done and not horizon_reached(step, args.max_episode_steps):
            state, obs = flatten_states(state_res, state_com, state_ind)
            actions = agent.act(obs, noise_std=noise_std)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
                env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
            )
            next_state, next_obs = flatten_states(next_res, next_com, next_ind)
            rewards = scalarized_rewards(obj_r, obj_c, obj_i, weight, args.reward_scale)
            replay.add(state, obs, actions, rewards, next_state, next_obs, done)
            episode_objectives += np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1
            global_step += 1
            if replay.size >= args.batch_size and global_step % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    agent.update(replay, args.batch_size)

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
                "noise_std": noise_std,
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
                "format": "matd3_state_dict_v1",
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
    parser = argparse.ArgumentParser(description="Train a fixed-weight MATD3 baseline.")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "baseline_matd3")
    parser.add_argument("--num-weights", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=162)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--noise-min", type=float, default=0.03)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
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
    if args.policy_delay < 1:
        raise ValueError("--policy-delay must be at least 1")
    save_dir = args.save_dir if args.save_dir.is_absolute() else PROJECT_ROOT / args.save_dir
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["save_dir"] = str(save_dir)
    config["algorithm"] = "fixed_weight_matd3"
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
        "algorithm": "fixed_weight_matd3",
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
