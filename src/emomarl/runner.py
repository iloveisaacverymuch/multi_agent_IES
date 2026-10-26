import os
import sys
import csv
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import random
from src.emomarl.sample import Sample
from src.emomarl.task import Task
from src.emomarl.ep import EP
from src.emomarl.population import Population
from src.emomarl.opt_graph import OptGraph
from src.emomarl.warm_up import initialize_warm_up_batch
from src.emomarl.mopg import MOPG_worker
from copy import deepcopy
from multiprocessing import Process, Queue, Event
import time
import queue
from src.emomarl.arguments import get_parser
from src.utils import get_ep_indices


def select_diverse_masac_tasks(args, ep, population):
    source = ep.sample_batch if len(ep.sample_batch) > 0 else population.sample_batch
    source = [
        sample for sample in source
        if getattr(sample, 'optgraph_id', None) is not None
        and getattr(sample, 'optgraph_id', None) >= 0
    ]
    if len(source) == 0:
        source = [
            sample for sample in population.sample_batch
            if getattr(sample, 'optgraph_id', None) is not None
            and getattr(sample, 'optgraph_id', None) >= 0
        ]
    if len(source) == 0:
        return [], [], []

    objs = np.array([sample.objs for sample in source])
    front_indices = get_ep_indices(objs)
    front_samples = [source[i] for i in front_indices]
    front_samples = sorted(front_samples, key=lambda sample: sample.objs[0])

    task_count = min(args.num_tasks, len(front_samples))
    if task_count == 0:
        return [], [], []

    if len(front_samples) == 1:
        selected_samples = [front_samples[0] for _ in range(args.num_tasks)]
    else:
        positions = np.linspace(0, len(front_samples) - 1, args.num_tasks)
        selected_samples = [front_samples[int(round(pos))] for pos in positions]

    if args.obj_num == 2:
        if args.num_tasks <= 1:
            weight_batch = [np.array([0.5, 0.5])]
        else:
            cost_weights = np.linspace(1.0, 0.0, args.num_tasks)
            weight_batch = [
                np.array([cost_w, 1.0 - cost_w], dtype=float)
                for cost_w in cost_weights
            ]
    else:
        weight_batch = [np.ones(args.obj_num, dtype=float) / args.obj_num for _ in selected_samples]

    predicted = [deepcopy(sample.objs) for sample in selected_samples]
    return selected_samples, weight_batch, predicted


def save_final_ep(args, ep):
    final_dir = Path(args.save_dir) / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)

    for old_policy in final_dir.glob('EP_policy_*.pt'):
        old_policy.unlink()

    for i, sample in enumerate(ep.sample_batch):
        torch.save(sample.actor, final_dir / f'EP_policy_{i}.pt')

    with open(final_dir / 'objs.txt', 'w') as fp:
        for obj in ep.obj_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

    metadata_path = final_dir / 'metadata.csv'
    try:
        metadata_fp = open(metadata_path, 'w', newline='')
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_path = final_dir / f'metadata_{timestamp}.csv'
        print(f"WARNING: metadata.csv is locked; writing metadata to {metadata_path}")
        metadata_fp = open(metadata_path, 'w', newline='')

    with metadata_fp as fp:
        writer = csv.writer(fp)
        writer.writerow([
            'pareto_index',
            'policy_file',
            'cost',
            'carbon',
            'weight_cost',
            'weight_carbon',
            'task_id',
            'generation',
            'update_step',
            'optgraph_id',
            'parent_optgraph_id',
        ])
        for i, (obj, sample) in enumerate(zip(ep.obj_batch, ep.sample_batch)):
            weight = getattr(sample, 'source_weight', None)
            if weight is None:
                weight_cost = ''
                weight_carbon = ''
            else:
                weight_cost = weight[0] if len(weight) > 0 else ''
                weight_carbon = weight[1] if len(weight) > 1 else ''
            writer.writerow([
                i,
                f'EP_policy_{i}.pt',
                obj[0],
                obj[1],
                weight_cost,
                weight_carbon,
                getattr(sample, 'source_task_id', ''),
                getattr(sample, 'source_generation', ''),
                getattr(sample, 'source_update_step', ''),
                getattr(sample, 'optgraph_id', ''),
                getattr(sample, 'parent_optgraph_id', ''),
            ])


def write_episode_reward_log(args, generation, samples):
    rows = []
    for sample in samples:
        if getattr(sample, 'episode_reward', None) is None:
            continue
        weight = getattr(sample, 'source_weight', None)
        agent_rewards = getattr(sample, 'episode_agent_rewards', None)
        episode_objs = getattr(sample, 'episode_objs', None)
        eval_objs = getattr(sample, 'objs', None)
        rows.append({
            'generation': generation,
            'task_id': getattr(sample, 'source_task_id', ''),
            'update_step': getattr(sample, 'source_update_step', ''),
            'weight_cost': weight[0] if weight is not None and len(weight) > 0 else '',
            'weight_carbon': weight[1] if weight is not None and len(weight) > 1 else '',
            'episode_reward': getattr(sample, 'episode_reward', ''),
            'reward_residential': agent_rewards[0] if agent_rewards is not None else '',
            'reward_commercial': agent_rewards[1] if agent_rewards is not None else '',
            'reward_industry': agent_rewards[2] if agent_rewards is not None else '',
            'episode_cost': episode_objs[0] if episode_objs is not None else '',
            'episode_carbon': episode_objs[1] if episode_objs is not None else '',
            'eval_cost': eval_objs[0] if eval_objs is not None else '',
            'eval_carbon': eval_objs[1] if eval_objs is not None else '',
        })
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    generation_path = Path(args.save_dir) / str(generation) / 'episode_rewards.csv'
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    with generation_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = Path(args.save_dir) / 'episode_rewards.csv'
    write_header = not aggregate_path.exists()
    with aggregate_path.open('a', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run(args):
    run_start_time = time.time()
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    args.save_dir = str(save_dir.resolve())

    # 设置随机种子，确保实验结果可复现
    np.random.seed(args.seed)  # NumPy随机数种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # PyTorch随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if (
        getattr(args, 'algorithm', 'mappo') in (
            'conditioned_masac',
            'decomposed_conditioned_masac',
        )
        and getattr(args, 'conditioned_centralized_learner', True)
    ):
        from src.emomarl.centralized_runner import run_conditioned_centralized
        return run_conditioned_centralized(args)

    total_generation = args.total_generation

    # 初始化核心数据结构
    ep = EP()  # 外部帕累托存档（存储所有非支配解）
    population = Population(args)
    opt_graph = OptGraph()  # 优化图（记录策略的优化历史）

    # 生成预热阶段的初始策略和权重
    # elite_batch: 初始策略列表
    # weight_batch: 对应的权重列表（均匀分布，如[1,0], [0.5,0.5], [0,1]）
    elite_batch, weight_batch = initialize_warm_up_batch(args)
    fixed_weight_batch = [deepcopy(weight) for weight in weight_batch]

    # 将初始解插入优化图，为每个策略分配唯一的节点ID
    for sample, weight in zip(elite_batch, weight_batch):
        sample.optgraph_id = opt_graph.insert(
            deepcopy(weight),  # 优化时使用的权重向量
            deepcopy(sample.objs),  # 策略的多目标回报值
            -1  # 初始节点无父节点（-1表示根节点）
        )

    # 初始化循环计数器
    generation_now = 0  # 迭代轮次计数器（用于区分热身/进化阶段）
    while generation_now < total_generation:
        # 打印阶段标识
        if generation_now == 0:
            num_updates = args.warmup_iter
            print('\n------------------------------- 热身阶段 -------------------------------')
        else:
            num_updates = args.update_iter
            print(f'\n-------------------- 进化阶段: 第 {generation_now} 代 --------------------')
        generation_now += 1

        # 构建任务批次：将选中的策略和权重打包成任务列表
        # 每个任务代表一个“策略-权重”对，将在独立的进程中优化
        task_batch = []
        for elite, weight in zip(elite_batch, weight_batch):
            task_batch.append(Task(elite, weight))

        processes = []
        results_queue = Queue()
        done_event = Event()

        # 启动子进程
        for task_id, task in enumerate(task_batch):
            p = Process(
                target=MOPG_worker,
                args=(args, task_id, task, num_updates, results_queue, done_event, generation_now)
            )
            p.daemon = True
            p.start()
            processes.append(p)

        # 结果容器
        all_offspring_batch = [[] for _ in range(len(processes))]
        cnt_done_workers = 0

        # ===================== 收集结果（确保所有子进程跑完）=====================
        while cnt_done_workers < len(processes):
            try:
                # 【关键】超时设长一些，避免子进程还在跑就误判
                rl_results = results_queue.get(timeout=10)

                task_id = rl_results['task_id']
                offsprings = rl_results['offspring_batch']

                # 检查是否有错误
                if 'error' in rl_results:
                    print(f"Worker {task_id} error: {rl_results['error']}")
                    cnt_done_workers += 1
                    continue

                # 拷贝样本
                for sample in offsprings:
                    all_offspring_batch[task_id].append(Sample.copy_from(sample))

                # 【关键】只有收到 done=True，才算该worker真正完成
                if rl_results['done']:
                    cnt_done_workers += 1
                    print(f"Worker {task_id} completed all iterations.")

            except queue.Empty:
                # 检查进程是否还活着
                alive_count = sum(1 for p in processes if p.is_alive())
                # dead_count = len(processes) - alive_count

                print(f"Queue empty. Alive: {alive_count}, Done workers: {cnt_done_workers}/{len(processes)}")

                # 只有所有进程都死了，才退出
                if alive_count == 0 and cnt_done_workers < len(processes):
                    print("ERROR: All workers died before completing!")
                    break

                # 还有进程活着，继续等待（可能只是暂时没产出）
                time.sleep(1)

        print(f"All workers completed. Collected results from {cnt_done_workers}/{len(processes)} workers.")

        # ===================== 安全回收进程 =====================
        # 【关键】所有子进程都跑完了，现在通知它们可以退出
        done_event.set()

        # 等待子进程自行退出
        for i, p in enumerate(processes):
            p.join(timeout=30)
            if p.is_alive():
                print(f"Force terminating worker {i}...")
                p.terminate()
                p.join(timeout=5)

        # 整理子代数据，分为三个用途
        all_sample_batch = []  # 所有中间策略（用于EP更新）
        last_offspring_batch = [None] * len(processes)  # 每个任务的最后一个策略（用于RA/PFA）
        offspring_batch = []  # 定期采样的策略（用于种群更新）

        for task_id in range(len(processes)):
            offsprings = all_offspring_batch[task_id]
            if len(offsprings) == 0:
                print(f"WARNING: task {task_id} produced no offspring; skipping it.")
                continue
            prev_node_id = task_batch[task_id].sample.optgraph_id   # 父节点ID
            opt_weights = deepcopy(task_batch[task_id].weight)

            for i, sample in enumerate(offsprings):
                # 把每个task中所有策略都加入all_sample_batch（用于EP更新）
                all_sample_batch.append(sample)

                # 每隔update_iter步采样一次（用于种群更新）
                # 例如：update_iter=5时，只记录第5、10、15...个策略
                if (i + 1) % args.sample_iter == 0:
                    # 将策略插入优化图，记录父子关系
                    prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)

            # 保存每个任务的最后一个策略（用于RA/PFA等基线方法）
            last_offspring_batch[task_id] = offsprings[-1]

        # 通知所有子进程任务结束（设置done_event标志）
        done_event.set()

        # -------------------- 2.2 更新存档 --------------------
        # 更新外部帕累托存档：只保留所有非支配解
        ep.update(all_sample_batch)
        write_episode_reward_log(args, generation_now, all_sample_batch)

        # 更新种群：通过性能缓冲区筛选，保持多样性和规模
        population.update(offspring_batch)

        # -------------------- 2.3 任务选择策略 --------------------
        # 根据配置选择不同的任务选择方法，决定下一轮要优化的策略和权重
        if generation_now == total_generation:
            elite_batch, weight_batch, predicted_offspring_objs = [], [], []
            print('Final generation completed; skipping next-task selection.')
        elif getattr(args, 'algorithm', 'mappo') == 'fixed_weight_mappo':
            next_elite_batch = []
            for task_id, sample in enumerate(last_offspring_batch):
                if sample is None:
                    sample = task_batch[task_id].sample
                sample.source_weight = deepcopy(fixed_weight_batch[task_id])
                if getattr(sample, 'optgraph_id', None) is None or sample.optgraph_id < 0:
                    sample.optgraph_id = opt_graph.insert(
                        deepcopy(fixed_weight_batch[task_id]),
                        deepcopy(sample.objs),
                        task_batch[task_id].sample.optgraph_id,
                    )
                next_elite_batch.append(sample)
            elite_batch = next_elite_batch
            weight_batch = [deepcopy(weight) for weight in fixed_weight_batch]
            predicted_offspring_objs = [deepcopy(sample.objs) for sample in elite_batch]
        elif (
            getattr(args, 'algorithm', 'mappo') in ('masac', 'conditioned_masac')
            and args.masac_diverse_task_selection
        ):
            elite_batch, weight_batch, predicted_offspring_objs = select_diverse_masac_tasks(
                args, ep, population
            )
        else:
            elite_batch, weight_batch, predicted_offspring_objs = \
                population.prediction_guided_selection(args, ep, opt_graph)

        # 打印本轮选中的任务信息
        if generation_now != total_generation:
            print('Selected Tasks:')
            for i in range(len(elite_batch)):
                print('objs = {}, weight = {}'.format(elite_batch[i].objs, weight_batch[i]))


        # -------------------- 2.4 保存结果 --------------------
        # 保存EP（外部帕累托存档）的目标值
        ep_dir = os.path.join(args.save_dir, str(generation_now), 'ep')
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # 保存种群的目标值
        population_dir = os.path.join(args.save_dir, str(generation_now), 'population')
        os.makedirs(population_dir, exist_ok=True)
        with open(os.path.join(population_dir, 'objs.txt'), 'w') as fp:
            for sample in population.sample_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(sample.objs)))


        # 保存优化图（记录所有策略的权重、目标值和父节点关系）
        with open(os.path.join(population_dir, 'optgraph.txt'), 'w') as fp:
            # 第一行：节点总数
            fp.write('{}\n'.format(len(opt_graph.objs)))
            # 每一行：权重向量; 目标值向量; 父节点ID
            for i in range(len(opt_graph.objs)):
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';' +
                          '{:5f}' + (args.obj_num - 1) * ',{:5f}' + ';{}\n')
                         .format(*(opt_graph.weights[i]), *(opt_graph.objs[i]), opt_graph.prev[i]))
            # 种群中每个策略对应的优化图节点ID
            fp.write('{}\n'.format(len(population.sample_batch)))
            for sample in population.sample_batch:
                fp.write('{}\n'.format(sample.optgraph_id))

        # 保存选中的精英策略的目标值和对应权重
        elite_dir = os.path.join(args.save_dir, str(generation_now), 'elites')
        os.makedirs(elite_dir, exist_ok=True)

        # 保存精英策略的目标值
        with open(os.path.join(elite_dir, 'elites.txt'), 'w') as fp:
            for elite in elite_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(elite.objs)))

        # 保存精英策略对应的权重
        with open(os.path.join(elite_dir, 'weights.txt'), 'w') as fp:
            for weight in weight_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(weight)))

        # 如果是预测引导方法，额外保存预测的改进目标值
        if args.selection_method in (
            'prediction-guided',
            'contribution-guided',
            'coverage-contribution-guided',
            'uncertainty-contribution-guided',
            'ensemble-uncertainty-guided',
            'ensemble-uncertainty-extreme-guided',
            'hv-ensemble-guided',
        ):
            with open(os.path.join(elite_dir, 'predictions.txt'), 'w') as fp:
                for objs in predicted_offspring_objs:
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(objs)))

        # 保存所有子代策略的目标值
        with open(os.path.join(elite_dir, 'offsprings.txt'), 'w') as fp:
            for i in range(len(all_offspring_batch)):
                for j in range(len(all_offspring_batch[i])):
                    fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n')
                             .format(*(all_offspring_batch[i][j].objs)))

        if generation_now == total_generation:
            save_final_ep(args, ep)
            with open(os.path.join(args.save_dir, 'elapsed_seconds.txt'), 'w') as fp:
                fp.write(f'{time.time() - run_start_time:.3f}\n')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
