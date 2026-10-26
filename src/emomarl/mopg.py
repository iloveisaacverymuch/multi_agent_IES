import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from collections import deque  # 双端队列，用于存储最近的结果
from copy import deepcopy  # 深拷贝，避免数据污染
import random
import torch  # PyTorch深度学习框架
from src.envs.ies_env import env as IESEnv
from src.emomarl.sample import Sample  # 自定义类，存储策略样本及其评估结果
from src.emomarl.conditioned_masac import ConditionedMASACReplayBuffer
from src.emomarl.masac import MASACReplayBuffer
from src.emomarl.replay_buffer import ReplayBuffer
from src.emomarl.reward_utils import scalarized_training_reward
import time
from multiprocessing import Queue, Event


def make_ies_env(args=None):
    if args is None:
        return IESEnv()
    return IESEnv(
        market_mechanism=getattr(args, "market_mechanism", "standard"),
        cda_carbon_price=getattr(args, "cda_carbon_price", 0.05),
        cda_dynamic_carbon_price=getattr(args, "cda_dynamic_carbon_price", False),
        cda_carbon_pressure_scale=getattr(args, "cda_carbon_pressure_scale", 1.0),
        cda_carbon_intensity_ref=getattr(args, "cda_carbon_intensity_ref", 1.0),
        cda_carbon_price_min_multiplier=getattr(args, "cda_carbon_price_min_multiplier", 0.5),
        cda_carbon_price_max_multiplier=getattr(args, "cda_carbon_price_max_multiplier", 3.0),
    )

#评估样本策略
def evaluation(sample, env_instance=None, args=None):
    env_instance = env_instance or make_ies_env(args)
    step = 0
    done = False
    action_set = sample.actor
    try:
        action_device = next(action_set[0].parameters()).device
    except StopIteration:
        action_device = torch.device("cpu")
    preference = getattr(sample, 'source_weight', None)
    if preference is not None:
        preference = torch.tensor(preference, dtype=torch.float32, device=action_device)
    objs_total = np.zeros(2)

    state_res, state_com, state_ind = env_instance.reset_train()

    with torch.no_grad():
        while not done:
            # 状态转tensor（全部在CPU，避免CUDA冲突）
            state_res_tensor = torch.tensor(
                state_res, dtype=torch.float32, device=action_device
            ).squeeze()
            state_com_tensor = torch.tensor(
                state_com, dtype=torch.float32, device=action_device
            ).squeeze()
            state_ind_tensor = torch.tensor(
                state_ind, dtype=torch.float32, device=action_device
            ).squeeze()

            # 动作推理
            if preference is not None and hasattr(action_set[0], 'default_preference'):
                action_res, mean_res, _, _ = action_set[0](
                    state_res_tensor, preference=preference, evaluation=True
                )
                action_com, mean_com, _, _ = action_set[1](
                    state_com_tensor, preference=preference, evaluation=True
                )
                action_ind, mean_ind, _, _ = action_set[2](
                    state_ind_tensor, preference=preference, evaluation=True
                )
            else:
                action_res, mean_res, _, _ = action_set[0](state_res_tensor, evaluation=True)
                action_com, mean_com, _, _ = action_set[1](state_com_tensor, evaluation=True)
                action_ind, mean_ind, _, _ = action_set[2](state_ind_tensor, evaluation=True)

            # 动作后处理
            action_res = torch.tanh(mean_res).reshape(1, -1)
            action_com = torch.tanh(mean_com).reshape(1, -1)
            action_ind = torch.tanh(mean_ind)
            action_ind = torch.cat([
                action_ind,
                torch.tensor([0.0], dtype=torch.float32, device=action_device),
            ]).reshape(1, -1)
            action = torch.cat((action_res, action_com, action_ind), dim=0)


            # 环境步进（必须传numpy）
            state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = env_instance.step(
                action.detach().cpu().numpy(), state_np, step, flag=0
            )

            # 累加目标
            objs_total += (obj_r + obj_c + obj_i)

            # 更新状态
            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1

    schedule_res = env_instance.schedule_result_residential()
    schedule_com = env_instance.schedule_result_commercial()
    schedule_ind = env_instance.schedule_result_industry()

    return objs_total, schedule_res, schedule_com, schedule_ind


def MASAC_worker(args, task_id, task, update_iter, results_queue, done_event, generation=None):
    agent = task.sample.agent
    actor_dict = agent.actor_dict
    w = task.weight
    env_instance = make_ies_env(args)
    worker_seed = args.seed + task_id * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    replay_buffer = MASACReplayBuffer(args)
    offspring_batch = []
    env_steps = 0

    for j in range(update_iter):
        state_res, state_com, state_ind = env_instance.reset_train()
        done = 0
        step = 0

        while not done:
            state_res_t = torch.tensor(state_res, dtype=torch.float32).squeeze()
            state_com_t = torch.tensor(state_com, dtype=torch.float32).squeeze()
            state_ind_t = torch.tensor(state_ind, dtype=torch.float32).squeeze()

            if env_steps < args.masac_random_steps:
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
                action_res, _, _, _ = actor_dict[0](state_res_t, evaluation=False)
                action_com, _, _, _ = actor_dict[1](state_com_t, evaluation=False)
                action_ind, _, _, _ = actor_dict[2](state_ind_t, evaluation=False)
                if args.masac_action_noise > 0:
                    decay = max(
                        args.masac_noise_min_scale,
                        1.0 - env_steps / max(args.masac_noise_decay_steps, 1),
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
            action_ind_env = torch.cat([action_ind.view(-1), torch.tensor([0.0])], dim=0).view(1, -1)
            action_env = torch.cat((action_res_env, action_com_env, action_ind_env), dim=0)

            state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = env_instance.step(
                action_env.detach().numpy(), state_np, step, flag=0
            )

            rewards = [
                scalarized_training_reward(obj_r, w, args),
                scalarized_training_reward(obj_c, w, args),
                scalarized_training_reward(obj_i, w, args),
            ]
            replay_buffer.store(
                states=[state_res, state_com, state_ind],
                actions=[
                    action_res.detach().numpy(),
                    action_com.detach().numpy(),
                    action_ind.detach().numpy(),
                ],
                rewards=rewards,
                next_states=[next_res, next_com, next_ind],
                done=done,
            )

            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1
            env_steps += 1

            if (
                len(replay_buffer) >= args.masac_start_steps
                and args.masac_update_every > 0
                and env_steps % args.masac_update_every == 0
            ):
                agent.update(replay_buffer, updates=1)

        if len(replay_buffer) >= args.masac_start_steps:
            agent.update(replay_buffer, updates=args.masac_updates_per_iter)

        sample = Sample(
            deepcopy(agent),
            deepcopy(agent.actor_dict),
            deepcopy(agent.critic),
            source_weight=deepcopy(w),
            source_task_id=task_id,
            source_generation=generation,
            source_update_step=j + 1,
            parent_optgraph_id=task.sample.optgraph_id,
        )
        objs, _, _, _ = evaluation(sample, env_instance=env_instance)
        sample.objs = objs
        offspring_batch.append(sample)

        if (j + 1) % args.ppo_iter == 0 or j == update_iter - 1:
            results_queue.put({
                'task_id': task_id,
                'offspring_batch': offspring_batch,
                'done': (j == update_iter - 1)
            })
            offspring_batch = []

    print(f"Worker {task_id}: all {update_iter} MASAC iterations completed.")
    print(f"Worker {task_id}: waiting for main process signal...")
    while not done_event.is_set():
        time.sleep(0.5)

    print(f"Worker {task_id}: exiting.")


def ConditionedMASAC_worker(args, task_id, task, update_iter, results_queue, done_event, generation=None):
    agent = task.sample.agent
    actor_dict = agent.actor_dict
    w = np.asarray(task.weight, dtype=np.float32)
    w = w / max(float(w.sum()), 1e-8)
    w_tensor = torch.tensor(w, dtype=torch.float32)
    if hasattr(agent, "set_default_preference"):
        agent.set_default_preference(w)

    env_instance = make_ies_env(args)
    worker_seed = args.seed + task_id * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    replay_buffer = ConditionedMASACReplayBuffer(args)
    offspring_batch = []
    env_steps = 0

    for j in range(update_iter):
        state_res, state_com, state_ind = env_instance.reset_train()
        done = 0
        step = 0

        while not done:
            state_res_t = torch.tensor(state_res, dtype=torch.float32).squeeze()
            state_com_t = torch.tensor(state_com, dtype=torch.float32).squeeze()
            state_ind_t = torch.tensor(state_ind, dtype=torch.float32).squeeze()

            if env_steps < args.masac_random_steps:
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
                    state_res_t, preference=w_tensor, evaluation=False
                )
                action_com, _, _, _ = actor_dict[1](
                    state_com_t, preference=w_tensor, evaluation=False
                )
                action_ind, _, _, _ = actor_dict[2](
                    state_ind_t, preference=w_tensor, evaluation=False
                )
                if args.masac_action_noise > 0:
                    decay = max(
                        args.masac_noise_min_scale,
                        1.0 - env_steps / max(args.masac_noise_decay_steps, 1),
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
            action_ind_env = torch.cat([action_ind.view(-1), torch.tensor([0.0])], dim=0).view(1, -1)
            action_env = torch.cat((action_res_env, action_com_env, action_ind_env), dim=0)

            state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = env_instance.step(
                action_env.detach().numpy(), state_np, step, flag=0
            )

            rewards = [
                scalarized_training_reward(obj_r, w, args),
                scalarized_training_reward(obj_c, w, args),
                scalarized_training_reward(obj_i, w, args),
            ]
            replay_buffer.store(
                states=[state_res, state_com, state_ind],
                actions=[
                    action_res.detach().numpy(),
                    action_com.detach().numpy(),
                    action_ind.detach().numpy(),
                ],
                rewards=rewards,
                reward_objs=[obj_r, obj_c, obj_i],
                next_states=[next_res, next_com, next_ind],
                done=done,
                preference=w,
            )

            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1
            env_steps += 1

            if (
                len(replay_buffer) >= args.masac_start_steps
                and args.masac_update_every > 0
                and env_steps % args.masac_update_every == 0
            ):
                agent.update(replay_buffer, updates=1)

        if len(replay_buffer) >= args.masac_start_steps:
            agent.update(replay_buffer, updates=args.masac_updates_per_iter)

        agent.set_default_preference(w)
        sample = Sample(
            deepcopy(agent),
            deepcopy(agent.actor_dict),
            deepcopy(agent.critic),
            source_weight=deepcopy(w),
            source_task_id=task_id,
            source_generation=generation,
            source_update_step=j + 1,
            parent_optgraph_id=task.sample.optgraph_id,
        )
        objs, _, _, _ = evaluation(sample, env_instance=env_instance)
        sample.objs = objs
        offspring_batch.append(sample)

        if (j + 1) % args.ppo_iter == 0 or j == update_iter - 1:
            results_queue.put({
                'task_id': task_id,
                'offspring_batch': offspring_batch,
                'done': (j == update_iter - 1)
            })
            offspring_batch = []

    print(f"Worker {task_id}: all {update_iter} Conditioned MASAC iterations completed.")
    print(f"Worker {task_id}: waiting for main process signal...")
    while not done_event.is_set():
        time.sleep(0.5)

    print(f"Worker {task_id}: exiting.")


def MOPG_worker(args, task_id, task, update_iter, results_queue, done_event, generation=None):
    """
    多进程工作器（确保跑完所有轮次版）
    """
    if getattr(args, 'algorithm', 'mappo') == 'conditioned_masac':
        return ConditionedMASAC_worker(args, task_id, task, update_iter, results_queue, done_event, generation)

    if getattr(args, 'algorithm', 'mappo') == 'masac':
        return MASAC_worker(args, task_id, task, update_iter, results_queue, done_event, generation)

    actor_dict = task.sample.actor
    critic_dict = task.sample.critic
    agent = task.sample.agent
    w = task.weight
    env_instance = make_ies_env(args)
    worker_seed = args.seed + task_id * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    replay_buffer = ReplayBuffer(args)
    offspring_batch = []

        # ===================== 主循环：必须跑完所有 update_iter =====================
    for j in range(update_iter):

        # 【可选】每轮开始时检查是否需要提前退出
        # 注意：如果任务必须跑完，可以注释掉这行
        # if done_event.is_set():
        #     print(f"Worker {task_id}: early exit at iter {j}")
        #     break

        state_res, state_com, state_ind = env_instance.reset_train()
        done = 0
        step = 0
        num_buffer = 0
        episode_agent_rewards = np.zeros(3, dtype=np.float64)
        episode_objs = np.zeros(2, dtype=np.float64)

        while not done:
            s_np = np.concatenate((state_res, state_com, state_ind), axis=0)
            s_tensor = torch.tensor(s_np, dtype=torch.float32)

            state_res_t = torch.tensor(state_res, dtype=torch.float32).squeeze()
            state_com_t = torch.tensor(state_com, dtype=torch.float32).squeeze()
            state_ind_t = torch.tensor(state_ind, dtype=torch.float32).squeeze()

            # 动作
            action_res, mean_res, _, log_prob_res = actor_dict[0](state_res_t, evaluation=False)
            action_com, mean_com, _, log_prob_com = actor_dict[1](state_com_t, evaluation=False)
            action_ind, mean_ind, _, log_prob_ind = actor_dict[2](state_ind_t, evaluation=False)

            action_res = action_res.view(1, -1)
            action_com = action_com.view(1, -1)
            action_ind = torch.cat([action_ind.view(-1), torch.tensor([0])], dim=0).view(1, -1)
            action = torch.cat((action_res, action_com, action_ind), dim=0)


            # 价值：centralized critic 为每个智能体分别输出一个 V_i(s)
            values = agent.get_value(s_tensor.flatten())

            # 环境交互
            action = action.detach()
            next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = env_instance.step(
                action.numpy(), s_np, step, flag=0
            )

            # 奖励
            reward_r = scalarized_training_reward(obj_r, w, args)
            reward_c = scalarized_training_reward(obj_c, w, args)
            reward_i = scalarized_training_reward(obj_i, w, args)
            episode_agent_rewards += np.array([reward_r, reward_c, reward_i], dtype=np.float64)
            episode_objs += obj_r + obj_c + obj_i

            # 数据转到CPU
            values = values.detach()
            logp_r = log_prob_res.detach()
            logp_c = log_prob_com.detach()
            logp_i = log_prob_ind.detach()

            act_r = action_res.detach().squeeze()
            act_c = action_com.detach().squeeze()
            act_i = action_ind.detach().squeeze()[:4]

            # 存经验
            replay_buffer.store_transition(num_buffer, 0, state_res_t, act_r, reward_r, next_res, values[0], logp_r, done)
            replay_buffer.store_transition(num_buffer, 1, state_com_t, act_c, reward_c, next_com, values[1], logp_c, done)
            replay_buffer.store_transition(num_buffer, 2, state_ind_t, act_i, reward_i, next_ind, values[2], logp_i, done)
            num_buffer += 1

            # 更新状态
            state_res, state_com, state_ind = next_res, next_com, next_ind
            step += 1

            # PPO 更新
            if replay_buffer.buffer_len() >= args.buffer_size:
                agent.update(replay_buffer.get_data())
                replay_buffer.clear()
                num_buffer = 0

        if replay_buffer.buffer_len() > 0:
            agent.update(replay_buffer.get_data())
            replay_buffer.clear()

        # 生成样本
        sample = Sample(
            deepcopy(agent),
            deepcopy(actor_dict),
            deepcopy(critic_dict),
            source_weight=deepcopy(w),
            source_task_id=task_id,
            source_generation=generation,
            source_update_step=j + 1,
            parent_optgraph_id=task.sample.optgraph_id,
            episode_reward=float(np.sum(episode_agent_rewards)),
            episode_agent_rewards=episode_agent_rewards,
            episode_objs=episode_objs,
        )
        objs, _, _, _ = evaluation(sample, env_instance=env_instance)
        sample.objs = objs
        offspring_batch.append(sample)

        # 发送中间结果
        if (j + 1) % args.ppo_iter == 0 or j == update_iter - 1:
            results_queue.put({
                'task_id': task_id,
                'offspring_batch': offspring_batch,
                'done': (j == update_iter - 1)  # ← 只有最后一轮才发 done=True
            })
            offspring_batch = []

    # ===================== 所有轮次跑完了！=====================
    print(f"Worker {task_id}: all {update_iter} iterations completed.")


    # ===================== 【关键】等待主进程确认后再退出 =====================
    # 主进程会在收集完所有结果后设置 done_event
    print(f"Worker {task_id}: waiting for main process signal...")
    while not done_event.is_set():
        time.sleep(0.5)

    print(f"Worker {task_id}: exiting.")
