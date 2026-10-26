import queue
import random
import time
from copy import deepcopy
from multiprocessing import Event, Process, Queue

import numpy as np
import torch

from src.envs.ies_env import env as IESEnv
from src.emomarl.conditioned_masac import ConditionedMASACReplayBuffer
from src.emomarl.mopg import evaluation, make_ies_env
from src.emomarl.reward_utils import scalarized_training_reward
from src.emomarl.sample import Sample


def _normalize_weight(weight):
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    return weight / max(float(weight.sum()), 1e-8)


def _agent_to_cpu_sample(agent, weight, task_id, generation, update_step, parent_optgraph_id):
    cpu_agent = deepcopy(agent)
    if hasattr(cpu_agent, "to_device"):
        cpu_agent.to_device("cpu")
    if hasattr(cpu_agent, "set_default_preference"):
        cpu_agent.set_default_preference(weight)
    return Sample(
        cpu_agent,
        deepcopy(cpu_agent.actor_dict),
        deepcopy(cpu_agent.critic),
        source_weight=deepcopy(weight),
        source_task_id=task_id,
        source_generation=generation,
        source_update_step=update_step,
        parent_optgraph_id=parent_optgraph_id,
    )


def _select_central_base_task(task_batch):
    if len(task_batch) == 1:
        return task_batch[0]
    uniform = np.ones_like(np.asarray(task_batch[0].weight, dtype=np.float32))
    uniform = uniform / uniform.sum()
    return min(
        task_batch,
        key=lambda task: np.linalg.norm(_normalize_weight(task.weight) - uniform),
    )


def build_dense_preference_specs(args, task_batch):
    if getattr(args, "conditioned_eval_grid_size", 0) <= 0:
        weights = [_normalize_weight(task.weight) for task in task_batch]
    elif args.obj_num == 2:
        grid_size = max(2, int(args.conditioned_eval_grid_size))
        cost_weights = np.linspace(1.0, 0.0, grid_size)
        weights = [
            np.array([cost_w, 1.0 - cost_w], dtype=np.float32)
            for cost_w in cost_weights
        ]
    else:
        weights = [_normalize_weight(task.weight) for task in task_batch]

    task_weights = [_normalize_weight(task.weight) for task in task_batch]
    specs = []
    for eval_id, weight in enumerate(weights):
        nearest_task_id = int(np.argmin([
            np.linalg.norm(weight - task_weight)
            for task_weight in task_weights
        ]))
        parent_task = task_batch[nearest_task_id]
        specs.append({
            "eval_id": eval_id,
            "nearest_task_id": nearest_task_id,
            "weight": weight,
            "parent_optgraph_id": parent_task.sample.optgraph_id,
        })
    return specs


def conditioned_sampler_worker(
    args,
    task_id,
    actor_dict,
    weight,
    env_steps_start,
    result_queue,
    done_event,
):
    env_instance = make_ies_env(args)
    worker_seed = args.seed + task_id * 1000 + int(env_steps_start)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    for actor in actor_dict.values():
        actor.cpu()
        actor.eval()

    weight = _normalize_weight(weight)
    weight_tensor = torch.tensor(weight, dtype=torch.float32)
    transitions = []
    objs_total = np.zeros(args.obj_num, dtype=np.float64)

    state_res, state_com, state_ind = env_instance.reset_train()
    done = 0
    step = 0
    local_steps = 0

    try:
        with torch.no_grad():
            while not done:
                state_res_t = torch.tensor(state_res, dtype=torch.float32).squeeze()
                state_com_t = torch.tensor(state_com, dtype=torch.float32).squeeze()
                state_ind_t = torch.tensor(state_ind, dtype=torch.float32).squeeze()

                current_env_step = env_steps_start + local_steps
                if current_env_step < args.masac_random_steps:
                    action_res = torch.tensor(
                        np.random.uniform(-1.0, 1.0, args.action_dim_per_agent[0]),
                        dtype=torch.float32,
                    )
                    action_com = torch.tensor(
                        np.random.uniform(-1.0, 1.0, args.action_dim_per_agent[1]),
                        dtype=torch.float32,
                    )
                    action_ind = torch.tensor(
                        np.random.uniform(-1.0, 1.0, args.action_dim_per_agent[2]),
                        dtype=torch.float32,
                    )
                else:
                    action_res, _, _, _ = actor_dict[0](
                        state_res_t, preference=weight_tensor, evaluation=False
                    )
                    action_com, _, _, _ = actor_dict[1](
                        state_com_t, preference=weight_tensor, evaluation=False
                    )
                    action_ind, _, _, _ = actor_dict[2](
                        state_ind_t, preference=weight_tensor, evaluation=False
                    )
                    if args.masac_action_noise > 0:
                        decay = max(
                            args.masac_noise_min_scale,
                            1.0 - current_env_step / max(args.masac_noise_decay_steps, 1),
                        )
                        noise_std = args.masac_action_noise * decay
                        action_res = action_res + torch.randn_like(action_res) * noise_std
                        action_com = action_com + torch.randn_like(action_com) * noise_std
                        action_ind = action_ind + torch.randn_like(action_ind) * noise_std

                action_res = torch.clamp(action_res, -0.999, 0.999)
                action_com = torch.clamp(action_com, -0.999, 0.999)
                action_ind = torch.clamp(action_ind, -0.999, 0.999)

                action_res_env = action_res.view(1, -1)
                action_com_env = action_com.view(1, -1)
                action_ind_env = torch.cat(
                    [action_ind.view(-1), torch.tensor([0.0])], dim=0
                ).view(1, -1)
                action_env = torch.cat((action_res_env, action_com_env, action_ind_env), dim=0)

                state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
                next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = env_instance.step(
                    action_env.numpy(), state_np, step, flag=0
                )

                rewards = [
                    scalarized_training_reward(obj_r, weight, args),
                    scalarized_training_reward(obj_c, weight, args),
                    scalarized_training_reward(obj_i, weight, args),
                ]
                transitions.append({
                    "states": [state_res, state_com, state_ind],
                    "actions": [
                        action_res.numpy(),
                        action_com.numpy(),
                        action_ind.numpy(),
                    ],
                    "rewards": rewards,
                    "reward_objs": [obj_r, obj_c, obj_i],
                    "next_states": [next_res, next_com, next_ind],
                    "done": done,
                    "preference": weight,
                })
                objs_total += obj_r + obj_c + obj_i

                state_res, state_com, state_ind = next_res, next_com, next_ind
                step += 1
                local_steps += 1

        result_queue.put({
            "task_id": task_id,
            "transitions": transitions,
            "steps": local_steps,
            "episode_objs": objs_total,
            "done": True,
        })
    except Exception as exc:
        result_queue.put({
            "task_id": task_id,
            "transitions": [],
            "steps": local_steps,
            "error": repr(exc),
            "done": True,
        })

    while not done_event.is_set():
        time.sleep(0.1)


def _collect_one_episode_per_task(args, task_batch, actor_dict, env_steps):
    result_queue = Queue()
    done_event = Event()
    processes = []

    actor_snapshot = deepcopy(actor_dict)
    for actor in actor_snapshot.values():
        actor.cpu()

    for task_id, task in enumerate(task_batch):
        p = Process(
            target=conditioned_sampler_worker,
            args=(
                args,
                task_id,
                deepcopy(actor_snapshot),
                deepcopy(task.weight),
                env_steps,
                result_queue,
                done_event,
            ),
        )
        p.daemon = True
        p.start()
        processes.append(p)

    results = {}
    while len(results) < len(processes):
        try:
            result = result_queue.get(timeout=10)
            results[result["task_id"]] = result
            if "error" in result:
                print(f"Sampler {result['task_id']} error: {result['error']}")
            else:
                print(
                    f"Sampler {result['task_id']} collected "
                    f"{len(result['transitions'])} transitions."
                )
        except queue.Empty:
            alive_count = sum(1 for p in processes if p.is_alive())
            print(
                f"Central sampler queue empty. Alive: {alive_count}, "
                f"Done samplers: {len(results)}/{len(processes)}"
            )
            if alive_count == 0:
                break

    done_event.set()
    for i, p in enumerate(processes):
        p.join(timeout=30)
        if p.is_alive():
            print(f"Force terminating sampler {i}...")
            p.terminate()
            p.join(timeout=5)

    return results


def initialize_shared_conditioned_learner(args, sample_batch):
    if len(sample_batch) == 0:
        raise ValueError("Cannot initialize Conditioned MASAC learner from an empty sample batch.")
    uniform = np.ones(args.obj_num, dtype=np.float32) / args.obj_num
    base_sample = min(
        sample_batch,
        key=lambda sample: np.linalg.norm(
            _normalize_weight(getattr(sample, "source_weight", uniform)) - uniform
        ),
    )
    agent = deepcopy(base_sample.agent)
    if hasattr(agent, "to_device"):
        agent.to_device(getattr(args, "device", "auto"))
    return agent


def _evaluate_preference_sample(args, agent, replay_buffer, spec, generation, update_step, eval_id):
    weight = _normalize_weight(spec["weight"])
    specialist_updates = int(getattr(args, "conditioned_specialist_updates", 0))
    if specialist_updates > 0 and len(replay_buffer) >= args.masac_start_steps:
        eval_agent = deepcopy(agent)
        if hasattr(eval_agent, "to_device"):
            eval_agent.to_device(getattr(args, "device", "auto"))
        if hasattr(eval_agent, "set_default_preference"):
            eval_agent.set_default_preference(weight)
        eval_agent.update(
            replay_buffer,
            updates=specialist_updates,
            preference_override=weight,
        )
    else:
        eval_agent = agent

    sample = _agent_to_cpu_sample(
        eval_agent,
        weight,
        spec.get("eval_id", eval_id),
        generation,
        update_step,
        spec.get("parent_optgraph_id"),
    )
    objs, _, _, _ = evaluation(sample, args=args)
    sample.objs = objs
    return sample


def run_conditioned_centralized_generation(
    args,
    task_batch,
    update_iter,
    generation,
    agent=None,
    replay_buffer=None,
    env_steps=0,
    eval_specs=None,
):
    if agent is None:
        base_task = _select_central_base_task(task_batch)
        agent = deepcopy(base_task.sample.agent)
        if hasattr(agent, "to_device"):
            agent.to_device(getattr(args, "device", "auto"))

    if replay_buffer is None:
        replay_buffer = ConditionedMASACReplayBuffer(args)
    if eval_specs is None:
        eval_specs = build_dense_preference_specs(args, task_batch)

    all_offspring_batch = [[] for _ in eval_specs]
    algorithm_name = getattr(args, "algorithm", "conditioned_masac")
    print(f"{algorithm_name} centralized learner device: {agent.device}")

    for update_step in range(update_iter):
        if hasattr(agent, "to_device"):
            agent.to_device(getattr(args, "device", "auto"))
        results = _collect_one_episode_per_task(args, task_batch, agent.actor_dict, env_steps)

        collected_steps = 0
        for task_id in range(len(task_batch)):
            result = results.get(task_id)
            if result is None or "error" in result:
                continue
            collected_steps += result["steps"]
            for transition in result["transitions"]:
                replay_buffer.store(
                    states=transition["states"],
                    actions=transition["actions"],
                    rewards=transition["rewards"],
                    reward_objs=transition.get("reward_objs"),
                    next_states=transition["next_states"],
                    done=transition["done"],
                    preference=transition["preference"],
                )
        env_steps += collected_steps

        if len(replay_buffer) >= args.masac_start_steps:
            agent.update(replay_buffer, updates=args.masac_updates_per_iter)

        for eval_id, spec in enumerate(eval_specs):
            sample = _evaluate_preference_sample(
                args,
                agent,
                replay_buffer,
                spec,
                generation,
                update_step + 1,
                eval_id,
            )
            all_offspring_batch[eval_id].append(sample)

        print(
            f"Central learner step {update_step + 1}/{update_iter}: "
            f"buffer={len(replay_buffer)}, env_steps={env_steps}"
        )

    return agent, replay_buffer, env_steps, all_offspring_batch, eval_specs
