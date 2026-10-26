import itertools
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from pymoo.indicators.hv import Hypervolume
from copy import deepcopy
#生成权重向量
def generate_w_batch_test(reward_dim, step_size):
    mesh_array = []
    step_size = step_size
    for i in range(reward_dim):
        mesh_array.append(np.arange(0, 1 + step_size, step_size))

    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1, :]
    w_batch_test = np.unique(w_batch_test, axis=0)

    return w_batch_test

def check_dominated_minimize(obj_batch, obj):
    """
    检查 obj 是否被 obj_batch 中的任何解支配（最小化语境）

    参数:
        obj_batch: (n, m) 数组，n个解，每个有m个目标
        obj: (m,) 数组，待检查的解

    返回:
        True 如果 obj 被支配，否则 False
    """
    # 找到所有在各方面都不差于 obj 的解
    not_worse = (obj_batch <= obj).all(axis=1)
    # 在这些解中，是否有严格优于 obj 的（至少一个目标更小）
    strictly_better = (obj_batch < obj).any(axis=1)
    # 两者同时满足 → 支配关系成立
    return (not_worse & strictly_better).any()


def pareto_front_minimize(obj_batch):
    """
    返回最小化问题下的帕累托前沿索引

    参数:
        obj_batch: (n, m) 数组

    返回:
        前沿索引数组
    """
    if len(obj_batch) == 0:
        return np.array([])

    obj_batch = np.asarray(obj_batch)
    n = len(obj_batch)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue

        for j in range(n):
            if i == j or is_dominated[j]:
                continue

            # 检查 i 是否支配 j（最小化）
            i_dom_j = (obj_batch[i] <= obj_batch[j]).all() and (obj_batch[i] < obj_batch[j]).any()
            # 检查 j 是否支配 i
            j_dom_i = (obj_batch[j] <= obj_batch[i]).all() and (obj_batch[j] < obj_batch[i]).any()

            if i_dom_j:
                is_dominated[j] = True
            elif j_dom_i:
                is_dominated[i] = True
                break  # i 被支配，无需继续比较

    return np.where(~is_dominated)[0]

def get_ep_indices(obj_batch_input):
    if len(obj_batch_input) == 0:
        return np.array([])
    obj_batch = np.array(obj_batch_input)
    return pareto_front_minimize(obj_batch)

#计算超体积
def compute_hypervolume(pareto_front):
    pareto_front = pareto_front
    ref_point = np.array([1e8, 1e8])   #参考点也转换成numpy数组
    hv_calculator = Hypervolume(ref_point=ref_point)
    hv = hv_calculator.do(pareto_front)
    return hv

#计算稀疏度
def compute_sparsity(pareto_front):
    N = pareto_front.shape[0]
    if N < 2:
         sparsity = 0.0  # 单个解时稀疏度为0
    else:
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
        diffs = np.diff(sorted_front, axis=0)  # 计算相邻解的差值
        squared_dists = np.sum(diffs ** 2, axis=1)  # 各维度差值的平方和
        sparsity = np.mean(squared_dists)
    return sparsity


def update_ep(ep_objs_batch, new_objs, eps=1e-5):
    """
    更新帕累托前沿（最小化问题）

    参数:
        ep_objs_batch: 现有前沿，二维 NumPy 数组，形状 (N, M)，按第一目标升序排列
        new_objs: 新解，一维 NumPy 数组，形状 (M,)
        eps: 浮点数容差

    返回:
        new_front: 更新后的前沿，二维 NumPy 数组，形状 (N', M)，按第一目标升序排列
    """
    # 确保输入是 NumPy 数组
    ep_objs_batch = np.asarray(ep_objs_batch)
    new_objs = np.asarray(new_objs)

    # 处理空前沿的情况
    if ep_objs_batch.shape[0] == 0:
        # 检查新解是否合法（所有目标 >= 0）
        if (new_objs < 0).any():
            return np.empty((0, new_objs.shape[0]))
        else:
            return new_objs.reshape(1, -1)

    # 负值过滤
    if (new_objs < 0).any():
        return ep_objs_batch.copy()

    # 使用列表收集保留的旧解（避免频繁数组拼接）
    keep_indices = []
    new_objs_should_add = True

    for i in range(ep_objs_batch.shape[0]):
        old = ep_objs_batch[i]

        # 判断 old 是否支配 new_objs
        # old <= new 在所有目标上，且至少一个严格小于
        old_dominates_new = (old <= new_objs + eps).all() and (old < new_objs - eps).any()

        # 判断 new_objs 是否支配 old
        new_dominates_old = (new_objs <= old + eps).all() and (new_objs < old - eps).any()

        if new_dominates_old:
            # new 支配 old，不保留 old，且 new 需要加入
            continue
        elif old_dominates_new:
            # old 支配 new，new 不需要加入
            new_objs_should_add = False
            keep_indices.append(i)
        else:
            # 互不支配，保留 old
            keep_indices.append(i)

    # 根据保留的索引构建新前沿
    new_front = ep_objs_batch[keep_indices].copy() if keep_indices else np.empty((0, ep_objs_batch.shape[1]))

    # 决定是否加入新解
    if new_objs_should_add:
        # 按第一目标排序插入
        if new_front.shape[0] == 0:
            new_front = new_objs.reshape(1, -1)
        else:
            # 找到插入位置
            insert_pos = np.searchsorted(new_front[:, 0], new_objs[0])
            new_front = np.insert(new_front, insert_pos, new_objs, axis=0)

    return new_front

#计算加入单个点后，外部帕累托前沿的变化
def update_ep_and_compute_hypervolume_sparsity(task_id, ep_objs_batch, new_objs, queue):
    new_ep_objs_batch = update_ep(ep_objs_batch, new_objs)
    hv = compute_hypervolume(new_ep_objs_batch)
    sparsity = compute_sparsity(new_ep_objs_batch)
    queue.put([task_id, hv, sparsity])

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


if __name__ == '__main__':
    a = generate_w_batch_test(2, 0.05)
    print(a)