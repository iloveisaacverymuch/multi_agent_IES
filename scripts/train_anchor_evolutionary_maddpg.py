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
    env_action,
    flatten_states,
    make_env,
)
from src.emomarl.opt_graph import OptGraph  # noqa: E402
from src.emomarl.population import predict_hyperbolic_ensemble  # noqa: E402


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def pareto_front(objectives):
    values = np.atleast_2d(np.asarray(objectives, dtype=np.float64))
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        keep[i] = not any(dominates(values[j], values[i]) for j in range(len(values)) if j != i)
    return values[keep]


def hypervolume_2d(front, reference=(1.1, 1.1)):
    values = pareto_front(front)
    values = values[np.argsort(values[:, 0])]
    area = 0.0
    previous_y = float(reference[1])
    for x, y in values:
        if x >= reference[0] or y >= previous_y:
            continue
        area += max(reference[0] - x, 0.0) * max(previous_y - y, 0.0)
        previous_y = y
    return float(area)


def normalized_hv_gain(front, prediction):
    front = np.atleast_2d(np.asarray(front, dtype=np.float64))
    prediction = np.asarray(prediction, dtype=np.float64)
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    scale = np.maximum(nadir - ideal, np.maximum(np.abs(ideal), 1.0) * 0.02)
    front_norm = np.clip((front - ideal) / scale, -0.1, 1.1)
    prediction_norm = np.clip((prediction - ideal) / scale, -0.1, 1.1)
    return max(
        0.0,
        hypervolume_2d(np.vstack([front_norm, prediction_norm])) - hypervolume_2d(front_norm),
    )


class VectorReplayBuffer:
    """Stores physical cost/carbon so samples can be relabelled after a preference change."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        state_dim = sum(OBS_DIMS)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = [np.zeros((capacity, dim), dtype=np.float32) for dim in OBS_DIMS]
        self.next_obs = [np.zeros((capacity, dim), dtype=np.float32) for dim in OBS_DIMS]
        self.actions = [np.zeros((capacity, dim), dtype=np.float32) for dim in ACT_DIMS]
        self.objectives = np.zeros((capacity, 3, 2), dtype=np.float32)
        self.stored_rewards = np.zeros((capacity, 3), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state,
        obs,
        actions,
        objectives,
        next_state,
        next_obs,
        done,
        collection_weight,
    ):
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        for agent_id in range(3):
            self.obs[agent_id][idx] = obs[agent_id]
            self.next_obs[agent_id][idx] = next_obs[agent_id]
            self.actions[agent_id][idx] = actions[agent_id]
        self.objectives[idx] = objectives
        self.stored_rewards[idx] = -np.sum(
            objectives * np.asarray(collection_weight, dtype=np.float32), axis=-1
        )
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        tensor = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=device)
        return {
            "states": tensor(self.states),
            "next_states": tensor(self.next_states),
            "obs": [tensor(values) for values in self.obs],
            "next_obs": [tensor(values) for values in self.next_obs],
            "actions": [tensor(values) for values in self.actions],
            "objectives": tensor(self.objectives),
            "stored_rewards": tensor(self.stored_rewards),
            "dones": tensor(self.dones),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


class RelabelledMADDPG:
    def __init__(self, args, seed):
        torch.manual_seed(seed)
        self.device = torch.device(args.device)
        self.gamma = args.gamma
        self.tau = args.tau
        self.reward_scale = args.reward_scale
        self.replay_mode = args.replay_mode
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
        self.actor_opts = [torch.optim.Adam(net.parameters(), lr=args.actor_lr) for net in self.actors]
        self.critic_opts = [torch.optim.Adam(net.parameters(), lr=args.critic_lr) for net in self.critics]

    def act(self, obs, rng, noise_std=0.0):
        actions = []
        with torch.no_grad():
            for agent_id, actor in enumerate(self.actors):
                value = actor(torch.as_tensor(obs[agent_id], dtype=torch.float32, device=self.device))
                action = value.cpu().numpy()
                if noise_std > 0.0:
                    action += rng.normal(0.0, noise_std, size=action.shape)
                actions.append(np.clip(action, -1.0, 1.0).astype(np.float32))
        return actions

    def update(self, replay, batch_size, weight, rng):
        batch = replay.sample(batch_size, self.device, rng)
        weight_tensor = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        if self.replay_mode == "scalar_stale":
            rewards = batch["stored_rewards"] * self.reward_scale
        else:
            rewards = -(batch["objectives"] * weight_tensor).sum(dim=-1) * self.reward_scale
        joint_actions = torch.cat(batch["actions"], dim=-1)
        with torch.no_grad():
            next_actions = [self.target_actors[i](batch["next_obs"][i]) for i in range(3)]
            next_joint_actions = torch.cat(next_actions, dim=-1)

        for agent_id in range(3):
            with torch.no_grad():
                target_q = self.target_critics[agent_id](batch["next_states"], next_joint_actions)
                target = rewards[:, agent_id : agent_id + 1] + self.gamma * (
                    1.0 - batch["dones"]
                ) * target_q
            q_value = self.critics[agent_id](batch["states"], joint_actions)
            loss = F.mse_loss(q_value, target)
            self.critic_opts[agent_id].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 1.0)
            self.critic_opts[agent_id].step()

        current_actions = [self.actors[i](batch["obs"][i]) for i in range(3)]
        for agent_id in range(3):
            actor_actions = [
                action if i == agent_id else action.detach()
                for i, action in enumerate(current_actions)
            ]
            loss = -self.critics[agent_id](
                batch["states"], torch.cat(actor_actions, dim=-1)
            ).mean()
            self.actor_opts[agent_id].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1.0)
            self.actor_opts[agent_id].step()
        self.soft_update()

    def soft_update(self):
        for source, target in zip(self.actors, self.target_actors):
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)
        for source, target in zip(self.critics, self.target_critics):
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.data.mul_(1.0 - self.tau).add_(self.tau * src.data)


class Task:
    def __init__(self, task_id, role, weight, args):
        self.task_id = task_id
        self.role = role
        self.weight = np.asarray(weight, dtype=np.float32)
        self.rng = np.random.default_rng(args.seed + task_id * 1000)
        self.agent = RelabelledMADDPG(args, args.seed + task_id * 1000)
        self.replay = VectorReplayBuffer(args.replay_size)
        self.episodes = 0
        self.global_step = 0
        self.graph_node = -1


def actor_snapshot(agent):
    return [
        {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        for actor in agent.actors
    ]


def update_archive(archive, objectives, task, generation):
    objectives = np.asarray(objectives, dtype=np.float64)
    if any(dominates(entry["objectives"], objectives) for entry in archive):
        return False
    rounded = np.round(objectives, 3)
    if any(np.array_equal(np.round(entry["objectives"], 3), rounded) for entry in archive):
        return False
    archive[:] = [entry for entry in archive if not dominates(objectives, entry["objectives"])]
    archive.append(
        {
            "objectives": objectives.copy(),
            "actor_state_dicts": actor_snapshot(task.agent),
            "task_id": task.task_id,
            "role": task.role,
            "generation": generation,
            "episode": task.episodes,
            "weight_cost": float(task.weight[0]),
            "weight_carbon": float(task.weight[1]),
        }
    )
    return True


def evaluate(task, args):
    environment = make_env(args)
    state_res, state_com, state_ind = environment.reset_train()
    done = False
    step = 0
    total = np.zeros(2, dtype=np.float64)
    while not done:
        _, obs = flatten_states(state_res, state_com, state_ind)
        actions = task.agent.act(obs, task.rng, noise_std=0.0)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
        )
        total += np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
        state_res, state_com, state_ind = next_res, next_com, next_ind
        step += 1
    return total


def train_episode(task, args, total_episodes_per_task):
    environment = make_env(args)
    state_res, state_com, state_ind = environment.reset_train()
    done = False
    step = 0
    total = np.zeros(2, dtype=np.float64)
    task.episodes += 1
    progress = task.episodes / max(total_episodes_per_task, 1)
    noise_std = max(args.noise_min, args.noise_std * (1.0 - progress))
    while not done:
        state, obs = flatten_states(state_res, state_com, state_ind)
        actions = task.agent.act(obs, task.rng, noise_std=noise_std)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            env_action(actions), np.vstack([state_res, state_com, state_ind]), step, flag=0
        )
        next_state, next_obs = flatten_states(next_res, next_com, next_ind)
        objective_vectors = np.asarray([obj_r, obj_c, obj_i], dtype=np.float32)
        task.replay.add(
            state,
            obs,
            actions,
            objective_vectors,
            next_state,
            next_obs,
            done,
            task.weight,
        )
        total += objective_vectors.sum(axis=0)
        state_res, state_com, state_ind = next_res, next_com, next_ind
        step += 1
        task.global_step += 1
        if task.replay.size >= args.batch_size and task.global_step % args.update_every == 0:
            for _ in range(args.updates_per_step):
                task.agent.update(task.replay, args.batch_size, task.weight, task.rng)
    return total, noise_std


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    result = np.zeros_like(values)
    if finite.any() and values[finite].max() > values[finite].min():
        result[finite] = (values[finite] - values[finite].min()) / (
            values[finite].max() - values[finite].min()
        )
    return result


def choose_evolving_weight(
    task, scope, tasks, archive, opt_graph, generation, args
):
    if scope == "low_tail":
        cost_values = np.linspace(0.0, args.explorer_tail, args.candidate_count)
        fallback = 0.0
    elif scope == "high_tail":
        cost_values = np.linspace(1.0 - args.explorer_tail, 1.0, args.candidate_count)
        fallback = 1.0
    else:
        cost_values = np.linspace(0.0, 1.0, args.candidate_count)
        cost_values = np.unique(np.append(cost_values, float(task.weight[0])))
        fallback = float(task.weight[0])
    candidates = np.column_stack([cost_values, 1.0 - cost_values])
    if args.explorer_selection == "random":
        selected = int(task.rng.integers(0, len(candidates)))
        return candidates[selected], {
            "mean_gain": 0.0,
            "uncertainty": 0.0,
            "score": 0.0,
            "reason": "random_preference",
        }
    if task.graph_node < 0 or len(opt_graph.objs) < 12 or len(archive) < 2:
        selected = int(np.argmin(np.abs(cost_values - fallback)))
        return candidates[selected], {
            "mean_gain": 0.0,
            "uncertainty": 0.0,
            "score": 0.0,
            "reason": "endpoint_warmup",
        }

    prediction = predict_hyperbolic_ensemble(args, opt_graph, task.graph_node, candidates.copy())
    ensemble = np.asarray(prediction["ensemble_predictions"], dtype=np.float64)
    archive_front = np.asarray([entry["objectives"] for entry in archive], dtype=np.float64)
    gains = np.zeros((ensemble.shape[0], len(candidates)), dtype=np.float64)
    for model_id in range(ensemble.shape[0]):
        for candidate_id in range(len(candidates)):
            gains[model_id, candidate_id] = normalized_hv_gain(
                archive_front, ensemble[model_id, candidate_id]
            )
    means = gains.mean(axis=0)
    stds = gains.std(axis=0)
    occupied = np.asarray([item.weight for item in tasks if item.task_id != task.task_id])
    coverage = np.min(
        np.linalg.norm(candidates[:, None, :] - occupied[None, :, :], axis=2), axis=1
    )
    switches = np.linalg.norm(candidates - task.weight[None, :], axis=1)
    progress = (generation - 1) / max(args.generations - 1, 1)
    kappa = args.kappa_start + progress * (args.kappa_end - args.kappa_start)
    score = (
        minmax(means)
        + kappa * minmax(stds)
        + args.coverage_weight * minmax(coverage)
        - args.switch_penalty * minmax(switches)
    )
    selected = int(np.argmax(score))
    if not np.isfinite(score[selected]) or np.allclose(score, score[0]):
        selected = int(np.argmin(np.abs(cost_values - fallback)))
    return candidates[selected], {
        "mean_gain": float(means[selected]),
        "uncertainty": float(stds[selected]),
        "score": float(score[selected]),
        "reason": "ensemble_hv_ucb",
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_archive(archive, args, save_dir):
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    archive.sort(key=lambda entry: entry["objectives"][0])
    metadata = []
    for index, entry in enumerate(archive):
        policy_file = f"EP_policy_{index}.pt"
        torch.save(
            {
                "format": "maddpg_state_dict_v1",
                "hidden_dim": args.hidden_dim,
                "actor_state_dicts": entry["actor_state_dicts"],
            },
            final_dir / policy_file,
        )
        metadata.append(
            {
                "pareto_index": index,
                "policy_file": policy_file,
                "cost": entry["objectives"][0],
                "carbon": entry["objectives"][1],
                "task_id": entry["task_id"],
                "role": entry["role"],
                "generation": entry["generation"],
                "episode": entry["episode"],
                "weight_cost": entry["weight_cost"],
                "weight_carbon": entry["weight_carbon"],
            }
        )
    np.savetxt(
        final_dir / "objs.txt",
        np.asarray([entry["objectives"] for entry in archive]),
        delimiter=",",
        fmt="%.6f",
    )
    write_csv(final_dir / "metadata.csv", metadata)
    retained = {row["policy_file"] for row in metadata}
    for policy_path in final_dir.glob("EP_policy_*.pt"):
        if policy_path.name not in retained:
            policy_path.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Carbon-Dividend CDA V3 + configurable Evolutionary MADDPG."
    )
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "anchor_evolutionary_maddpg")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--warmup-episodes", type=int, default=30)
    parser.add_argument("--generation-episodes", type=int, default=12)
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
    parser.add_argument(
        "--replay-mode",
        choices=["vector_relabel", "scalar_stale", "clear_on_change"],
        default="vector_relabel",
        help="Reward handling for dynamic preferences; the default preserves current behavior.",
    )
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--explorer-tail", type=float, default=0.35)
    parser.add_argument("--kappa-start", type=float, default=0.0)
    parser.add_argument("--kappa-end", type=float, default=0.0)
    parser.add_argument("--coverage-weight", type=float, default=0.08)
    parser.add_argument("--switch-penalty", type=float, default=0.03)
    parser.add_argument(
        "--explorer-selection",
        choices=["ensemble_hv_ucb", "random"],
        default="ensemble_hv_ucb",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["fixed", "evolving"],
        default="evolving",
    )
    parser.add_argument("--elite-ensemble-size", type=int, default=5)
    parser.add_argument("--elite-ensemble-sample-ratio", type=float, default=0.8)
    parser.add_argument("--elite-ensemble-max-samples", type=int, default=32)
    parser.add_argument("--obj-num", type=int, default=2)
    parser.add_argument(
        "--market-mechanism",
        default="carbon_dividend_v3",
        choices=["standard", "carbon_dividend_v3"],
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.save_dir = args.save_dir if args.save_dir.is_absolute() else PROJECT_ROOT / args.save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    anchor_cost_weights = [0.2, 0.4, 0.6, 0.8]
    anchor_role = "anchor" if args.anchor_mode == "fixed" else "evolving"
    task_specs = [(anchor_role, [value, 1.0 - value]) for value in anchor_cost_weights]
    if args.anchor_mode == "fixed":
        task_specs += [
            ("explorer_low_cost_weight", [0.0, 1.0]),
            ("explorer_high_cost_weight", [1.0, 0.0]),
        ]
    else:
        task_specs += [("evolving", [0.0, 1.0]), ("evolving", [1.0, 0.0])]
    tasks = [Task(index, role, weight, args) for index, (role, weight) in enumerate(task_specs)]
    total_episodes_per_task = args.warmup_episodes + (args.generations - 1) * args.generation_episodes
    expected_episodes = len(tasks) * total_episodes_per_task
    archive = []
    opt_graph = OptGraph()
    episode_rows = []
    selection_rows = []
    started = time.time()

    config = vars(args).copy()
    config["save_dir"] = str(args.save_dir)
    config["anchor_weights"] = anchor_cost_weights
    config["tasks"] = len(tasks)
    config["episodes_per_task"] = total_episodes_per_task
    config["total_training_episodes"] = expected_episodes
    config["total_deterministic_evaluations"] = expected_episodes
    (args.save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for generation in range(1, args.generations + 1):
        episodes_this_generation = args.warmup_episodes if generation == 1 else args.generation_episodes
        if generation > 1:
            if args.anchor_mode == "fixed":
                evolving_tasks = [task for task in tasks if task.role.startswith("explorer")]
                scopes = ["low_tail", "high_tail"]
            else:
                evolving_tasks = tasks
                scopes = ["full"] * len(tasks)
            for task, scope in zip(evolving_tasks, scopes):
                old_weight = task.weight.copy()
                new_weight, diagnostics = choose_evolving_weight(
                    task, scope, tasks, archive, opt_graph, generation, args
                )
                task.weight = np.asarray(new_weight, dtype=np.float32)
                if args.replay_mode == "clear_on_change" and not np.allclose(
                    old_weight, task.weight
                ):
                    task.replay.clear()
                selection_rows.append(
                    {
                        "generation": generation,
                        "task_id": task.task_id,
                        "old_weight_cost": float(old_weight[0]),
                        "new_weight_cost": float(task.weight[0]),
                        **diagnostics,
                    }
                )
                print(
                    f"generation {generation}: task {task.task_id} preference "
                    f"{old_weight[0]:.3f} -> {task.weight[0]:.3f} ({diagnostics['reason']})",
                    flush=True,
                )

        for task in tasks:
            for local_episode in range(1, episodes_this_generation + 1):
                train_objs, noise_std = train_episode(task, args, total_episodes_per_task)
                eval_objs = evaluate(task, args)
                previous_node = task.graph_node
                task.graph_node = opt_graph.insert(task.weight.copy(), eval_objs.copy(), previous_node)
                accepted = update_archive(archive, eval_objs, task, generation)
                episode_rows.append(
                    {
                        "generation": generation,
                        "task_id": task.task_id,
                        "role": task.role,
                        "task_episode": task.episodes,
                        "weight_cost": float(task.weight[0]),
                        "weight_carbon": float(task.weight[1]),
                        "noise_std": noise_std,
                        "train_cost": train_objs[0],
                        "train_carbon": train_objs[1],
                        "eval_cost": eval_objs[0],
                        "eval_carbon": eval_objs[1],
                        "archive_accepted": int(accepted),
                        "archive_size": len(archive),
                    }
                )
            print(
                f"generation {generation}/{args.generations}, task {task.task_id}, "
                f"episodes={task.episodes}/{total_episodes_per_task}, archive={len(archive)}",
                flush=True,
            )
        write_csv(args.save_dir / "episode_log.csv", episode_rows)
        write_csv(args.save_dir / "selection_history.csv", selection_rows)
        save_archive(archive, args, args.save_dir)

    elapsed = time.time() - started
    save_archive(archive, args, args.save_dir)
    write_csv(args.save_dir / "episode_log.csv", episode_rows)
    write_csv(args.save_dir / "selection_history.csv", selection_rows)
    (args.save_dir / "launcher_elapsed_seconds.txt").write_text(f"{elapsed:.6f}\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "seed": args.seed,
        "archive_points": len(archive),
        "training_episodes": len(episode_rows),
        "deterministic_evaluations": len(episode_rows),
        "elapsed_seconds": elapsed,
    }
    (args.save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
