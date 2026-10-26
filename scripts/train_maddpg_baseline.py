import argparse
import csv
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.ies_env import env as IESEnv


OBS_DIMS = [7, 7, 7]
ACT_DIMS = [5, 5, 4]


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def update_policy_archive(archive, objectives, agent, metadata):
    objectives = np.asarray(objectives, dtype=np.float64)
    if any(dominates(entry["objectives"], objectives) for entry in archive):
        return
    rounded = np.round(objectives, decimals=3)
    if any(np.array_equal(np.round(entry["objectives"], 3), rounded) for entry in archive):
        return
    archive[:] = [
        entry for entry in archive if not dominates(objectives, entry["objectives"])
    ]
    archive.append(
        {
            "objectives": objectives.copy(),
            "actors": {
                agent_id: deepcopy(actor).cpu().eval()
                for agent_id, actor in enumerate(agent.actors)
            },
            **metadata,
        }
    )


def make_weights(num_weights):
    if num_weights <= 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)
    cost_weights = np.linspace(0.0, 1.0, num_weights)
    return np.column_stack((cost_weights, 1.0 - cost_weights)).astype(np.float32)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, joint_action):
        return self.net(torch.cat([state, joint_action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, state_dim, obs_dims, act_dims):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = [np.zeros((capacity, dim), dtype=np.float32) for dim in obs_dims]
        self.next_obs = [np.zeros((capacity, dim), dtype=np.float32) for dim in obs_dims]
        self.actions = [np.zeros((capacity, dim), dtype=np.float32) for dim in act_dims]
        self.rewards = np.zeros((capacity, len(obs_dims)), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, obs, actions, rewards, next_state, next_obs, done):
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        for agent_id in range(len(self.obs)):
            self.obs[agent_id][idx] = obs[agent_id]
            self.next_obs[agent_id][idx] = next_obs[agent_id]
            self.actions[agent_id][idx] = actions[agent_id]
        self.rewards[idx] = rewards
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32, device=device),
            "next_states": torch.tensor(self.next_states[idx], dtype=torch.float32, device=device),
            "obs": [
                torch.tensor(agent_obs[idx], dtype=torch.float32, device=device)
                for agent_obs in self.obs
            ],
            "next_obs": [
                torch.tensor(agent_obs[idx], dtype=torch.float32, device=device)
                for agent_obs in self.next_obs
            ],
            "actions": [
                torch.tensor(agent_action[idx], dtype=torch.float32, device=device)
                for agent_action in self.actions
            ],
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device),
        }


class MADDPG:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.gamma = args.gamma
        self.tau = args.tau
        self.act_dims = ACT_DIMS
        self.actors = nn.ModuleList(
            [Actor(OBS_DIMS[i], ACT_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.target_actors = nn.ModuleList(
            [Actor(OBS_DIMS[i], ACT_DIMS[i], args.hidden_dim) for i in range(3)]
        ).to(self.device)
        self.critics = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)
        self.target_critics = nn.ModuleList(
            [Critic(sum(OBS_DIMS), sum(ACT_DIMS), args.hidden_dim) for _ in range(3)]
        ).to(self.device)
        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=args.actor_lr) for actor in self.actors
        ]
        self.critic_opts = [
            torch.optim.Adam(critic.parameters(), lr=args.critic_lr) for critic in self.critics
        ]

    def act(self, obs, noise_std=0.0):
        actions = []
        with torch.no_grad():
            for agent_id, actor in enumerate(self.actors):
                obs_t = torch.tensor(obs[agent_id], dtype=torch.float32, device=self.device)
                action = actor(obs_t).cpu().numpy()
                if noise_std > 0:
                    action = action + np.random.normal(0.0, noise_std, size=action.shape)
                actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
        return actions

    def update(self, replay, batch_size):
        batch = replay.sample(batch_size, self.device)
        joint_actions = torch.cat(batch["actions"], dim=-1)
        with torch.no_grad():
            next_actions = [
                self.target_actors[i](batch["next_obs"][i]) for i in range(3)
            ]
            next_joint_actions = torch.cat(next_actions, dim=-1)

        for agent_id in range(3):
            with torch.no_grad():
                target_q = self.target_critics[agent_id](
                    batch["next_states"], next_joint_actions
                )
                y = batch["rewards"][:, agent_id : agent_id + 1] + self.gamma * (
                    1.0 - batch["dones"]
                ) * target_q
            q = self.critics[agent_id](batch["states"], joint_actions)
            critic_loss = F.mse_loss(q, y)
            self.critic_opts[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 1.0)
            self.critic_opts[agent_id].step()

        current_actions = []
        for agent_id in range(3):
            current_actions.append(self.actors[agent_id](batch["obs"][agent_id]))
        for agent_id in range(3):
            actor_actions = [
                action if i == agent_id else action.detach()
                for i, action in enumerate(current_actions)
            ]
            actor_loss = -self.critics[agent_id](
                batch["states"], torch.cat(actor_actions, dim=-1)
            ).mean()
            self.actor_opts[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1.0)
            self.actor_opts[agent_id].step()

        self.soft_update()

    def soft_update(self):
        for source, target in zip(self.actors, self.target_actors):
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)
        for source, target in zip(self.critics, self.target_critics):
            for src_param, tgt_param in zip(source.parameters(), target.parameters()):
                tgt_param.data.mul_(1.0 - self.tau).add_(self.tau * src_param.data)


def make_env(args):
    return IESEnv(
        market_mechanism=args.market_mechanism,
        cda_carbon_price=args.cda_carbon_price,
    )


def flatten_states(state_res, state_com, state_ind):
    obs = [
        state_res.reshape(-1).astype(np.float32),
        state_com.reshape(-1).astype(np.float32),
        state_ind.reshape(-1).astype(np.float32),
    ]
    return np.concatenate(obs).astype(np.float32), obs


def env_action(actions):
    action_res = actions[0].reshape(1, -1)
    action_com = actions[1].reshape(1, -1)
    action_ind = np.concatenate([actions[2], np.array([0.0], dtype=np.float32)]).reshape(1, -1)
    return np.concatenate([action_res, action_com, action_ind], axis=0)


def scalarized_rewards(obj_r, obj_c, obj_i, weight, reward_scale):
    weight = weight / max(float(np.sum(weight)), 1e-8)
    objs = [obj_r, obj_c, obj_i]
    return np.array([-float(np.dot(obj, weight)) * reward_scale for obj in objs], dtype=np.float32)


def evaluate(agent, args):
    environment = make_env(args)
    state_res, state_com, state_ind = environment.reset_train()
    done = False
    step = 0
    total = np.zeros(2, dtype=np.float64)
    while not done:
        state, obs = flatten_states(state_res, state_com, state_ind)
        actions = agent.act(obs, noise_std=0.0)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
        )
        total += obj_r + obj_c + obj_i
        state_res, state_com, state_ind = next_res, next_com, next_ind
        step += 1
    return total


def train_one_weight(weight, weight_index, args, seed, policy_archive):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    agent = MADDPG(args)
    replay = ReplayBuffer(args.replay_size, sum(OBS_DIMS), OBS_DIMS, ACT_DIMS)
    episode_rows = []
    global_step = 0
    final_eval_objs = None

    for episode in range(1, args.episodes + 1):
        environment = make_env(args)
        state_res, state_com, state_ind = environment.reset_train()
        done = False
        step = 0
        episode_objs = np.zeros(2, dtype=np.float64)
        noise_std = max(args.noise_min, args.noise_std * (1.0 - episode / max(args.episodes, 1)))
        while not done:
            state, obs = flatten_states(state_res, state_com, state_ind)
            actions = agent.act(obs, noise_std=noise_std)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
                env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
            )
            next_state, next_obs = flatten_states(next_res, next_com, next_ind)
            rewards = scalarized_rewards(obj_r, obj_c, obj_i, weight, args.reward_scale)
            replay.add(state, obs, actions, rewards, next_state, next_obs, done)
            episode_objs += obj_r + obj_c + obj_i
            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1
            global_step += 1
            if replay.size >= args.batch_size and global_step % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    agent.update(replay, args.batch_size)
        should_evaluate = args.archive_every_episode or episode % args.eval_every == 0
        eval_objs = evaluate(agent, args) if should_evaluate else np.array([np.nan, np.nan])
        if should_evaluate:
            final_eval_objs = eval_objs
            if args.archive_every_episode:
                update_policy_archive(
                    policy_archive,
                    eval_objs,
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
                "train_cost": episode_objs[0],
                "train_carbon": episode_objs[1],
                "eval_cost": eval_objs[0],
                "eval_carbon": eval_objs[1],
            }
        )
    if final_eval_objs is None or not args.archive_every_episode:
        final_eval_objs = evaluate(agent, args)
    return agent, final_eval_objs, episode_rows


def main():
    parser = argparse.ArgumentParser(description="Train fixed-weight MADDPG baseline.")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "baseline_maddpg")
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
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--archive-every-episode", action="store_true")
    parser.add_argument(
        "--market-mechanism",
        choices=[
            "standard",
            "carbon_aware",
            "carbon_aware_v2",
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ],
        default="carbon_aware",
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.02)
    args = parser.parse_args()

    save_dir = args.save_dir if args.save_dir.is_absolute() else PROJECT_ROOT / args.save_dir
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    weights = make_weights(args.num_weights)
    policy_archive = []
    all_evaluations = []
    for idx, weight in enumerate(weights):
        agent, objs, rows = train_one_weight(
            weight, idx, args, args.seed + idx * 1000, policy_archive
        )
        if not args.archive_every_episode:
            policy_archive.append(
                {
                    "objectives": np.asarray(objs, dtype=np.float64).copy(),
                    "actors": {
                        agent_id: deepcopy(actor).cpu().eval()
                        for agent_id, actor in enumerate(agent.actors)
                    },
                    "weight_index": idx,
                    "episode": args.episodes,
                    "weight_cost": float(weight[0]),
                    "weight_carbon": float(weight[1]),
                }
            )
        all_evaluations.extend(
            [
                [row["eval_cost"], row["eval_carbon"]]
                for row in rows
                if np.isfinite(row["eval_cost"]) and np.isfinite(row["eval_carbon"])
            ]
        )
        pd_rows_path = save_dir / f"weight_{idx}_episodes.csv"
        with pd_rows_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"weight {idx}: w={weight.tolist()}, objs={objs.tolist()}")

    policy_archive.sort(key=lambda entry: entry["objectives"][0])
    objectives = np.asarray([entry["objectives"] for entry in policy_archive])
    metadata = []
    for pareto_index, entry in enumerate(policy_archive):
        policy_file = f"EP_policy_{pareto_index}.pt"
        if args.archive_every_episode:
            torch.save(
                {
                    "format": "maddpg_state_dict_v1",
                    "hidden_dim": args.hidden_dim,
                    "actor_state_dicts": [
                        actor.state_dict() for actor in entry["actors"].values()
                    ],
                },
                final_dir / policy_file,
            )
        else:
            torch.save(entry["actors"], final_dir / policy_file)
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

    with (final_dir / "metadata.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)
    np.savetxt(
        save_dir / "all_objs.txt",
        np.asarray(all_evaluations, dtype=np.float64),
        delimiter=",",
        fmt="%.6f",
    )
    print(f"Saved MADDPG baseline to {save_dir}")


if __name__ == "__main__":
    main()
