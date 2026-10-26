import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from src.emomarl.sample import Sample
from src.utils import get_ep_indices
from scipy.optimize import least_squares
from pymoo.indicators.hv import Hypervolume

#收集与目标节点相似的优化历史数据，用于训练预测模型
def collect_nearest_data(opt_graph, optgraph_id, threshold=0.1):
    """
    Args:
        opt_graph: 优化历史图对象
        optgraph_id: 目标节点ID
        threshold: 相似度阈值（相对目标值的百分比）
    Returns:
        objs_data: 相似节点的目标值列表
        weights_data: 子节点的归一化权重列表
        delta_objs_data: 子节点的目标改进量列表
    """
    objs_data, weights_data, delta_objs_data = [], [], []

    # 遍历所有节点
    for i in range(len(opt_graph.objs)):
        # 计算当前节点与目标节点的绝对差值
        diff = np.abs(opt_graph.objs[optgraph_id] - opt_graph.objs[i])

        # 检查所有目标维度是否都满足阈值条件
        if np.all(diff < np.abs(opt_graph.objs[optgraph_id]) * threshold):
            # 遍历当前节点的所有子节点
            for next_index in opt_graph.succ[i]:
                objs_data.append(opt_graph.objs[i])  # 记录父节点目标值
                # 归一化权重并存储
                weights_data.append(opt_graph.weights[next_index] / np.sum(opt_graph.weights[next_index]))
                delta_objs_data.append(opt_graph.delta_objs[next_index])  # 存储改进量

    return objs_data, weights_data, delta_objs_data

def predict_hyperbolic(args, opt_graph, optgraph_id, test_weights):
    """
    使用双曲函数预测给定权重下的目标改进量
    Args:
        args: 配置参数
        opt_graph: 优化历史图
        optgraph_id: 当前节点ID
        test_weights: 待测试的权重列表
    Returns:
        dict: 包含预测结果的字典
    """
    test_weights = np.array(test_weights)

    # 权重归一化（总和为1）
    for test_weight in test_weights:
        test_weight /= np.sum(test_weight)

    # 动态调整数据收集范围
    threshold, sigma = 0.1, 0.03
    for _ in range(20):
        # 收集相似数据
        #objs_data：策略的当前目标值 weights_data：优化时使用的权重 delta_objs_data：经过 RL 优化后的目标改进量
        objs_data, weights_data, delta_objs_data = collect_nearest_data(opt_graph, optgraph_id, threshold)

        # 检查唯一权重数量是否足够
        cnt_data = 0
        for i in range(len(weights_data)):
            if all(np.linalg.norm(weights_data[i] - weights_data[j]) >= 1e-5 for j in range(i)):
                cnt_data += 1
                if cnt_data > 3: break

        if cnt_data > 3:
            break
        else:  # 扩大搜索范围
            threshold *= 2.0
            sigma *= 2.0
    else:
        base_obj = deepcopy(opt_graph.objs[optgraph_id])
        return {
            'sample_index': optgraph_id,
            'predictions': [deepcopy(base_obj) for _ in test_weights]
        }

    # 定义双曲正切变换函数
    def f(x, A, a, b, c):
        """双曲预测函数：A * tanh(a(x-b)) + c"""
        return A * (np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1) + c

    # 最小二乘损失函数
    def fun(params, x, y):
        """带权重的残差计算"""
        A, a, b, c = params
        return (f(x, A, a, b, c) - y) * w  # 加权误差

    # 雅可比矩阵（加速优化）
    def jac(params, x, y):
        A, a, b, c = params[0], params[1], params[2], params[3]
        J = np.zeros([4, len(x)])
        J[0] = ((np.exp(a * (x - b)) - 1) / (np.exp(a * (x - b)) + 1)) * w  # df/dA
        J[1] = (A * (x - b) * 2 * np.exp(a * (x - b)) / (np.exp(a * (x - b)) + 1) ** 2) * w  # df/da
        J[2] = (A * (-a) * 2 * np.exp(a * (x - b)) / (np.exp(a * (x - b)) + 1) ** 2) * w  # df/db
        J[3] = w  # df/dc
        return J.T

    # 逐维度预测
    M = args.obj_num
    delta_predictions = []
    for dim in range(M):
        # 准备训练数据
        train_x = [w[dim] for w in weights_data]
        train_y = [d[dim] for d in delta_objs_data]

        # 计算数据点权重（高斯衰减）
        w = [np.exp(-(np.linalg.norm((o - opt_graph.objs[optgraph_id]) / opt_graph.objs[optgraph_id]) / sigma) ** 2 / 2)
             for o in objs_data]

        # 参数边界约束
        A_upperbound = np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0)
        bounds = ([0, 0.1, -5., -500.], [A_upperbound, 20., 5., 500.])

        # 鲁棒最小二乘拟合
        res = least_squares(fun, np.ones(4), loss='soft_l1',
                            bounds=bounds, args=(train_x, train_y), jac=jac)

        # 预测测试权重下的改进量
        delta_predictions.append(f(test_weights.T[dim], *res.x))

    # 合成最终预测结果
    predictions = [
        opt_graph.objs[optgraph_id] + delta
        for delta in np.array(delta_predictions).T
    ]

    return {'sample_index': optgraph_id, 'predictions': predictions}

#性能缓冲区将性能空间按角度划分为多个扇形区域，
# 每个区域只保留距离参考点最远的 pbuffer_size 个解，
# 从而在保持种群多样性的同时控制种群规模。
class _BootstrapOptGraph:
    def __init__(self, base_obj, objs_data, weights_data, delta_objs_data):
        self.objs = [deepcopy(base_obj)]
        self.weights = [np.ones_like(base_obj, dtype=float) / len(base_obj)]
        self.delta_objs = [np.zeros_like(base_obj, dtype=float)]
        self.succ = [[]]

        for obj, weight, delta in zip(objs_data, weights_data, delta_objs_data):
            parent_idx = len(self.objs)
            child_idx = parent_idx + 1
            self.objs.append(np.asarray(obj, dtype=float))
            self.weights.append(np.ones_like(base_obj, dtype=float) / len(base_obj))
            self.delta_objs.append(np.zeros_like(base_obj, dtype=float))
            self.succ.append([child_idx])

            self.objs.append(np.asarray(obj, dtype=float) + np.asarray(delta, dtype=float))
            normalized_weight = np.asarray(weight, dtype=float)
            normalized_weight = normalized_weight / np.sum(normalized_weight)
            self.weights.append(normalized_weight)
            self.delta_objs.append(np.asarray(delta, dtype=float))
            self.succ.append([])


def _collect_ensemble_data(opt_graph, optgraph_id):
    threshold = 0.1
    for _ in range(20):
        objs_data, weights_data, delta_objs_data = collect_nearest_data(opt_graph, optgraph_id, threshold)
        cnt_data = 0
        for i in range(len(weights_data)):
            if all(np.linalg.norm(weights_data[i] - weights_data[j]) >= 1e-5 for j in range(i)):
                cnt_data += 1
                if cnt_data > 3:
                    break
        if cnt_data > 3:
            return objs_data, weights_data, delta_objs_data
        threshold *= 2.0
    return [], [], []


def predict_hyperbolic_ensemble(args, opt_graph, optgraph_id, test_weights):
    test_weights = np.array(test_weights, dtype=float)
    for test_weight in test_weights:
        test_weight /= np.sum(test_weight)

    base_obj = deepcopy(opt_graph.objs[optgraph_id])
    objs_data, weights_data, delta_objs_data = _collect_ensemble_data(opt_graph, optgraph_id)
    if len(weights_data) == 0:
        base_predictions = [deepcopy(base_obj) for _ in test_weights]
        return {
            'sample_index': optgraph_id,
            'predictions': base_predictions,
            'prediction_std': [np.zeros(args.obj_num, dtype=float) for _ in test_weights],
            'uncertainty': [0.0 for _ in test_weights],
            'ensemble_predictions': np.repeat(
                np.asarray(base_predictions, dtype=float)[None, :, :],
                max(1, int(getattr(args, "elite_ensemble_size", 5))),
                axis=0,
            ),
        }

    ensemble_size = max(1, int(getattr(args, "elite_ensemble_size", 5)))
    sample_ratio = float(getattr(args, "elite_ensemble_sample_ratio", 0.80))
    sample_ratio = min(max(sample_ratio, 0.1), 1.5)
    rng_seed = int(getattr(args, "seed", 0)) + int(optgraph_id) * 1009 + len(opt_graph.objs) * 9176
    rng = np.random.default_rng(rng_seed)
    max_samples = int(getattr(args, "elite_ensemble_max_samples", 0))
    if max_samples > 0 and len(weights_data) > max_samples:
        pool_indices = rng.choice(len(weights_data), size=max_samples, replace=False)
        objs_data = [objs_data[i] for i in pool_indices]
        weights_data = [weights_data[i] for i in pool_indices]
        delta_objs_data = [delta_objs_data[i] for i in pool_indices]
    sample_size = max(4, int(round(len(weights_data) * sample_ratio)))

    ensemble_predictions = []
    for _ in range(ensemble_size):
        indices = rng.choice(len(weights_data), size=sample_size, replace=True)
        boot_graph = _BootstrapOptGraph(
            base_obj,
            [objs_data[i] for i in indices],
            [weights_data[i] for i in indices],
            [delta_objs_data[i] for i in indices],
        )
        try:
            result = predict_hyperbolic(args, boot_graph, 0, deepcopy(test_weights))
            ensemble_predictions.append(np.array(result['predictions'], dtype=float))
        except Exception:
            ensemble_predictions.append(np.array([deepcopy(base_obj) for _ in test_weights], dtype=float))

    ensemble_predictions = np.stack(ensemble_predictions, axis=0)
    prediction_mean = np.mean(ensemble_predictions, axis=0)
    prediction_std = np.std(ensemble_predictions, axis=0)
    scale = np.maximum(np.abs(base_obj), 1.0)
    uncertainty = np.linalg.norm(prediction_std / scale, axis=1)
    finite = uncertainty[np.isfinite(uncertainty)]
    if len(finite) > 0 and finite.max() > finite.min():
        uncertainty = (uncertainty - finite.min()) / (finite.max() - finite.min())
    else:
        uncertainty = np.zeros_like(uncertainty)

    return {
        'sample_index': optgraph_id,
        'predictions': [deepcopy(pred) for pred in prediction_mean],
        'prediction_std': [deepcopy(std) for std in prediction_std],
        'uncertainty': [float(value) for value in uncertainty],
        'ensemble_predictions': ensemble_predictions.copy(),
    }


class Population:
    """
    种群管理类
    负责维护策略种群，通过性能缓冲区（Performance Buffer）机制保持种群的多样性和质量
    """
    def __init__(self, args):
        """
        初始化种群管理器
        Args:
            args: 配置参数对象
        """
        self.sample_batch = []  # 种群中的所有策略样本
        self.pbuffer_num = args.pbuffer_num  # 性能缓冲区数量（如200）
        self.pbuffer_size = args.pbuffer_size  # 每个缓冲区的容量（如2）
        self.dtheta = np.pi / 2.0 / self.pbuffer_num  # 每个分区的角度宽度（弧度）
        self.z_min = np.zeros(args.obj_num)  # 当前种群的理想点（各目标最小值）
        self.pbuffers = None  # 存储每个分区中的样本索引
        self.pbuffer_dist = None  # 存储每个分区中样本的径向距离

    def insert_pbuffer(self, index, objs):
        """
        将策略插入性能缓冲区（按角度分区+距离排序）
        原理：将目标空间按极角划分为多个扇形区域，每个区域内按径向距离降序保留最优解

        Args:
            index: 策略在种群列表中的索引位置
            objs: 策略的目标值向量，如 [速度, 能耗]
        Returns:
            bool: 是否插入成功
        """
        # 计算相对参考点的偏移量
        f = objs - self.z_min  # 将目标空间原点平移到参考点

        if np.any(f < -1e-7):
            return False
        f = np.maximum(f, 0.0)

        # 极坐标转换
        dist = np.linalg.norm(f)  # 径向距离（到参考点的欧氏距离）
        if dist < 1e-12:
            theta = 0.0
        else:
            theta = np.arccos(np.clip(f[1] / dist, -1.0, 1.0))  # 极角（弧度）

        # 确定分区ID：根据极角除以分区宽度得到
        buffer_id = min(int(theta // self.dtheta), self.pbuffer_num - 1)
        # 验证分区编号是否有效
        if not 0 <= buffer_id < self.pbuffer_num:
            return False

        # 在对应分区中按距离升序插入（最小化问题中，越接近理想点越好）
        inserted = False
        for i in range(len(self.pbuffers[buffer_id])):
            # 找到第一个距离比当前解大的位置
            if self.pbuffer_dist[buffer_id][i] > dist:
                # 在该位置前插入（保持升序排列）
                self.pbuffers[buffer_id].insert(i, index)
                self.pbuffer_dist[buffer_id].insert(i, dist)
                inserted = True
                break

        # 维护缓冲区大小：如果超过容量，只保留前 pbuffer_size 个（距离最小的）
        if inserted and len(self.pbuffers[buffer_id]) > self.pbuffer_size:
            self.pbuffers[buffer_id] = self.pbuffers[buffer_id][:self.pbuffer_size]
            self.pbuffer_dist[buffer_id] = self.pbuffer_dist[buffer_id][:self.pbuffer_size]
        # 如果未插入（新解距离较大）但分区未满，则追加到末尾
        elif not inserted and len(self.pbuffers[buffer_id]) < self.pbuffer_size:
            self.pbuffers[buffer_id].append(index)
            self.pbuffer_dist[buffer_id].append(dist)
            inserted = True

        return inserted

    def update(self, sample_batch):
        """
        种群更新函数：将当前种群与子代种群合并，通过性能缓冲区筛选出最优个体，形成新一代种群

        Args:
            sample_batch: 子代样本列表
        """
        # 1. 合并当前种群和子代
        all_sample_batch = self.sample_batch + sample_batch
        if len(all_sample_batch) == 0:
            return
        self.z_min = np.min(np.array([sample.objs for sample in all_sample_batch]), axis=0)

        # 2. 清空现有数据结构，准备重建
        self.sample_batch = []  # 清空种群
        self.pbuffers = [[] for _ in range(self.pbuffer_num)]  # 清空性能缓冲区（存索引）
        self.pbuffer_dist = [[] for _ in range(self.pbuffer_num)]  # 清空距离缓冲区（存距离）

        # 3. 将所有个体插入性能缓冲区
        for i, sample in enumerate(all_sample_batch):
            self.insert_pbuffer(i, sample.objs)  # 按角度分区+距离排序

        # 4. 从缓冲区重建种群（按分区顺序收集保留的个体）
        for pbuffer in self.pbuffers:
            for index in pbuffer:
                self.sample_batch.append(all_sample_batch[index])

    def compute_hypervolume(self, objs_batch):
        """
        计算超体积（Hypervolume）指标
        超体积 = 帕累托前沿与参考点之间围成的面积（越大表示收敛性越好）

        Args:
            objs_batch: 目标值数组，形状 [n_samples, n_objectives]
        Returns:
            hv: 超体积值
        """
        # 提取非支配解（帕累托前沿）
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        if len(ep_objs_batch) == 0:
            return 0.0
        ref_point = np.array([1e8, 1e8])
        hv_indicator = Hypervolume(ref_point=ref_point)

        hypervolume = hv_indicator.do(ep_objs_batch)
        return hypervolume

    def compute_sparsity(self, objs_batch):
        """
        计算稀疏度（Sparsity）指标
        稀疏度 = 相邻帕累托点之间距离平方的平均值（越小表示分布越均匀）

        Args:
            objs_batch: 目标值数组
        Returns:
            sparsity: 稀疏度值
        """
        # 提取非支配解并排序
        ep_objs_batch = deepcopy(np.array(objs_batch)[get_ep_indices(objs_batch)])
        N = ep_objs_batch.shape[0]
        if N < 2:
            sparsity = 0.0  # 单个解时稀疏度为0
        else:
            sorted_front = ep_objs_batch[np.argsort(ep_objs_batch[:, 0])]
            diffs = np.diff(sorted_front, axis=0)  # 计算相邻解的差值
            squared_dists = np.sum(diffs ** 2, axis=1)  # 各维度差值的平方和
            sparsity = np.mean(squared_dists)
        return sparsity

    def _contribution_scores(self, args, candidates, mask, virtual_ep):
        base_hv = self.compute_hypervolume(virtual_ep)
        base_sparsity = self.compute_sparsity(virtual_ep)
        scores = []
        for i, candidate in enumerate(candidates):
            if not mask[i]:
                scores.append(-np.inf)
                continue
            candidate_ep = virtual_ep + [candidate['prediction']]
            hv_gain = self.compute_hypervolume(candidate_ep) - base_hv
            sparsity_gain = max(
                0.0,
                self.compute_sparsity(candidate_ep) - base_sparsity,
            )
            scores.append(hv_gain - args.sparsity * sparsity_gain)
        return scores

    def _distance_uncertainty_scores(self, args, candidates, mask, virtual_ep):
        reference_mode = getattr(args, "elite_uncertainty_reference", "ep_population")
        reference = []
        if reference_mode in ("ep", "ep_population"):
            reference.extend([np.asarray(obj, dtype=float) for obj in virtual_ep])
        if reference_mode in ("population", "ep_population"):
            reference.extend([np.asarray(sample.objs, dtype=float) for sample in self.sample_batch])

        selectable_predictions = [
            np.asarray(candidate["prediction"], dtype=float)
            for i, candidate in enumerate(candidates)
            if mask[i]
        ]
        if not selectable_predictions:
            return [-np.inf if not flag else 0.0 for flag in mask]
        if not reference:
            reference = selectable_predictions

        all_points = np.vstack(reference + selectable_predictions)
        mins = np.min(all_points, axis=0)
        ranges = np.max(all_points, axis=0) - mins
        ranges = np.where(ranges <= 1e-12, 1.0, ranges)
        reference_norm = (np.vstack(reference) - mins) / ranges

        uncertainties = []
        for i, candidate in enumerate(candidates):
            if not mask[i]:
                uncertainties.append(-np.inf)
                continue
            pred_norm = (np.asarray(candidate["prediction"], dtype=float) - mins) / ranges
            distances = np.linalg.norm(reference_norm - pred_norm, axis=1)
            uncertainties.append(float(np.min(distances)))

        finite = np.array([value for value in uncertainties if np.isfinite(value)], dtype=float)
        if len(finite) > 0 and finite.max() > finite.min():
            uncertainties = [
                (value - finite.min()) / (finite.max() - finite.min())
                if np.isfinite(value) else -np.inf
                for value in uncertainties
            ]
        else:
            uncertainties = [0.0 if np.isfinite(value) else -np.inf for value in uncertainties]
        return uncertainties

    def _uncertainty_contribution_scores(self, args, candidates, mask, virtual_ep):
        contribution = self._contribution_scores(args, candidates, mask, virtual_ep)
        uncertainty = self._distance_uncertainty_scores(args, candidates, mask, virtual_ep)

        finite_contribution = np.array(
            [score for score in contribution if np.isfinite(score)],
            dtype=float,
        )
        if len(finite_contribution) == 0:
            scale = 1.0
        else:
            span = finite_contribution.max() - finite_contribution.min()
            scale = max(float(span), float(np.max(np.abs(finite_contribution))), 1.0)

        kappa = max(0.0, float(getattr(args, "elite_uncertainty_kappa", 0.20)))
        scores = []
        for contribution_score, uncertainty_score in zip(contribution, uncertainty):
            if not np.isfinite(contribution_score):
                scores.append(-np.inf)
                continue
            scores.append(contribution_score + kappa * scale * uncertainty_score)
        return scores

    def _ensemble_uncertainty_scores(self, args, candidates, mask, virtual_ep):
        contribution = self._contribution_scores(args, candidates, mask, virtual_ep)
        raw_uncertainty = [
            float(candidate.get('uncertainty', 0.0)) if mask[i] else -np.inf
            for i, candidate in enumerate(candidates)
        ]
        finite_uncertainty = np.array(
            [value for value in raw_uncertainty if np.isfinite(value)],
            dtype=float,
        )
        if len(finite_uncertainty) > 0 and finite_uncertainty.max() > finite_uncertainty.min():
            uncertainty = [
                (value - finite_uncertainty.min()) / (finite_uncertainty.max() - finite_uncertainty.min())
                if np.isfinite(value) else -np.inf
                for value in raw_uncertainty
            ]
        else:
            uncertainty = [0.0 if np.isfinite(value) else -np.inf for value in raw_uncertainty]

        finite_contribution = np.array(
            [score for score in contribution if np.isfinite(score)],
            dtype=float,
        )
        if len(finite_contribution) == 0:
            scale = 1.0
        else:
            span = finite_contribution.max() - finite_contribution.min()
            scale = max(float(span), float(np.max(np.abs(finite_contribution))), 1.0)

        kappa = max(0.0, float(getattr(args, "elite_uncertainty_kappa", 0.20)))
        scores = []
        for contribution_score, uncertainty_score in zip(contribution, uncertainty):
            if not np.isfinite(contribution_score):
                scores.append(-np.inf)
                continue
            scores.append(contribution_score + kappa * scale * uncertainty_score)
        return scores

    @staticmethod
    def _hypervolume_gain_2d(front, prediction, ref_point=(1e8, 1e8)):
        """Return the exact 2-D HV gain of one point over a nondominated front."""
        prediction = np.asarray(prediction, dtype=float)
        reference = np.asarray(ref_point, dtype=float)
        if np.any(~np.isfinite(prediction)) or np.any(prediction >= reference):
            return 0.0
        front = np.asarray(front, dtype=float)
        if len(front) == 0:
            return float(np.prod(reference - prediction))
        if np.any(np.all(front <= prediction, axis=1)):
            return 0.0

        ordered = front[np.argsort(front[:, 0])]
        current_y = reference[1]
        for point in ordered:
            if point[0] <= prediction[0]:
                current_y = min(current_y, point[1])
        current_x = prediction[0]
        gain = 0.0
        for point in ordered:
            if point[0] <= prediction[0]:
                continue
            gain += max(0.0, point[0] - current_x) * max(
                0.0, current_y - prediction[1]
            )
            current_x = point[0]
            current_y = min(current_y, point[1])
            if current_y <= prediction[1]:
                break
        else:
            gain += max(0.0, reference[0] - current_x) * max(
                0.0, current_y - prediction[1]
            )
        return float(gain)

    def _hv_ensemble_scores(self, args, candidates, mask, virtual_ep):
        """Add an HV-space ensemble disagreement bonus to mean contribution."""
        contribution = self._contribution_scores(args, candidates, mask, virtual_ep)
        front = np.asarray(virtual_ep, dtype=float)
        front = front[get_ep_indices(front)] if len(front) else front
        disagreements = []
        for index, candidate in enumerate(candidates):
            if not mask[index]:
                disagreements.append(-np.inf)
                continue
            predictions = np.asarray(
                candidate.get("ensemble_predictions", [candidate["prediction"]]),
                dtype=float,
            )
            member_gains = [
                self._hypervolume_gain_2d(front, prediction) for prediction in predictions
            ]
            disagreements.append(float(np.std(member_gains)))

        finite_disagreement = np.asarray(
            [value for value in disagreements if np.isfinite(value)], dtype=float
        )
        if len(finite_disagreement) and finite_disagreement.max() > finite_disagreement.min():
            normalized_disagreement = [
                (value - finite_disagreement.min())
                / (finite_disagreement.max() - finite_disagreement.min())
                if np.isfinite(value)
                else -np.inf
                for value in disagreements
            ]
        else:
            normalized_disagreement = [
                0.0 if np.isfinite(value) else -np.inf for value in disagreements
            ]

        finite_contribution = np.asarray(
            [value for value in contribution if np.isfinite(value)], dtype=float
        )
        if len(finite_contribution):
            scale = max(
                float(finite_contribution.max() - finite_contribution.min()),
                float(np.max(np.abs(finite_contribution))),
                1.0,
            )
        else:
            scale = 1.0
        kappa = max(0.0, float(getattr(args, "elite_uncertainty_kappa", 0.10)))
        return [
            contribution_score + kappa * scale * disagreement
            if np.isfinite(contribution_score)
            else -np.inf
            for contribution_score, disagreement in zip(contribution, normalized_disagreement)
        ]

    @staticmethod
    def _select_extreme_or_best(candidates, mask, scores, selection_step, enabled=True):
        """Protect one cost and one carbon endpoint before score-only selection."""
        if enabled and selection_step < 2:
            objective_index = selection_step
            selectable = [index for index, available in enumerate(mask) if available]
            extreme_weight = max(
                candidates[index]["weight"][objective_index] for index in selectable
            )
            extreme_candidates = [
                index
                for index in selectable
                if candidates[index]["weight"][objective_index] >= extreme_weight - 1e-9
            ]
            return max(extreme_candidates, key=lambda index: scores[index])
        return int(np.argmax(scores))

    def _best_by_contribution(self, args, candidates, candidate_indices, virtual_ep):
        if not candidate_indices:
            return None
        base_hv = self.compute_hypervolume(virtual_ep)
        base_sparsity = self.compute_sparsity(virtual_ep)
        best_idx = candidate_indices[0]
        best_score = -np.inf
        for idx in candidate_indices:
            candidate_ep = virtual_ep + [candidates[idx]['prediction']]
            hv_gain = self.compute_hypervolume(candidate_ep) - base_hv
            sparsity_gain = max(
                0.0,
                self.compute_sparsity(candidate_ep) - base_sparsity,
            )
            score = hv_gain - args.sparsity * sparsity_gain
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _candidate_regions(self, candidates, bins):
        bins = max(1, int(bins))
        costs = np.array([candidate['prediction'][0] for candidate in candidates], dtype=float)
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        if max_cost - min_cost <= 1e-9:
            return [0 for _ in candidates]
        normalized = (costs - min_cost) / (max_cost - min_cost)
        return np.minimum((normalized * bins).astype(int), bins - 1).tolist()

    def _select_coverage_contribution_candidate(
        self,
        args,
        candidates,
        mask,
        virtual_ep,
        selected_regions,
        selected_extremes,
        selection_step,
    ):
        selectable = [i for i in range(len(candidates)) if mask[i]]
        if not selectable:
            return 0

        if getattr(args, "elite_extreme_protection", True):
            if "cost" not in selected_extremes:
                max_cost_weight = max(candidates[i]['weight'][0] for i in selectable)
                cost_candidates = [
                    i for i in selectable
                    if candidates[i]['weight'][0] >= max_cost_weight - 1e-9
                ]
                selected_extremes.add("cost")
                return self._best_by_contribution(args, candidates, cost_candidates, virtual_ep)

            if selection_step > 0 and "carbon" not in selected_extremes:
                max_carbon_weight = max(candidates[i]['weight'][1] for i in selectable)
                carbon_candidates = [
                    i for i in selectable
                    if candidates[i]['weight'][1] >= max_carbon_weight - 1e-9
                ]
                selected_extremes.add("carbon")
                return self._best_by_contribution(args, candidates, carbon_candidates, virtual_ep)

        regions = self._candidate_regions(
            candidates,
            getattr(args, "elite_coverage_bins", getattr(args, "num_tasks", 3)),
        )
        available_regions = sorted(set(regions[i] for i in selectable))
        region_counts = {
            region: selected_regions.count(region)
            for region in available_regions
        }
        target_count = min(region_counts.values())
        target_regions = {
            region for region, count in region_counts.items()
            if count == target_count
        }
        coverage_candidates = [
            i for i in selectable
            if regions[i] in target_regions
        ]
        return self._best_by_contribution(args, candidates, coverage_candidates, virtual_ep)

    # def evaluate_hv(self, candidates, mask, virtual_ep_objs_batch):
    #     """
    #     评估每个候选任务加入后虚拟存档的超体积
    #
    #     Args:
    #         candidates: 候选任务列表
    #         mask: 候选任务是否可选的掩码
    #         virtual_ep_objs_batch: 当前虚拟帕累托存档的目标值
    #     Returns:
    #         hv: 每个候选对应的超体积值列表
    #     """
    #     hv = [0.0 for _ in range(len(candidates))]
    #     for i in range(len(candidates)):
    #         if mask[i]:  # 只评估未被选中的候选
    #             # 构建加入预测点后的新目标集
    #             new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
    #             hv[i] = self.compute_hypervolume(new_objs_batch)
    #     return hv
    #
    # def evaluate_sparsity(self, candidates, mask, virtual_ep_objs_batch):
    #     """
    #     评估每个候选任务加入后虚拟存档的稀疏度
    #
    #     Args:
    #         candidates: 候选任务列表
    #         mask: 候选任务是否可选的掩码
    #         virtual_ep_objs_batch: 当前虚拟帕累托存档的目标值
    #     Returns:
    #         sparsity: 每个候选对应的稀疏度值列表
    #     """
    #     sparsity = [0.0 for _ in range(len(candidates))]
    #     for i in range(len(candidates)):
    #         if mask[i]:
    #             new_objs_batch = np.array(virtual_ep_objs_batch + [candidates[i]['prediction']])
    #             sparsity[i] = self.compute_sparsity(new_objs_batch)
    #     return sparsity

    def prediction_guided_selection(self, args, ep, opt_graph):
        """
        预测引导的任务选择算法（论文核心方法）
        贪心地选择最能提升帕累托前沿质量的 n 个“策略-权重”对

        Args:
            args: 配置参数
            iteration: 当前迭代次数
            ep: 外部帕累托存档
            opt_graph: 优化图（记录优化历史）
            w: 权重
        Returns:
            elite_batch: 选中的精英策略列表
            scalarization_batch: 对应的权重列表
            predicted_offspring_objs: 预测的目标值列表
        """
        candidates = []  # 存储所有候选任务

        # ========== 第一部分：为每个策略生成候选权重 ==========
        for sample in self.sample_batch:
            # 获取该策略在优化图中的权重中心（当前优化方向）
            weight_center = opt_graph.weights[sample.optgraph_id]
            # 计算权重向量在单位圆上的极角
            angle_center = np.arctan2(weight_center[1], weight_center[0])

            # 在 ±45° 范围内均匀生成测试权重
            test_weights = []
            for i in range(args.num_weight_candidates):
                # 角度从 中心-45° 到 中心+45° 均匀分布
                angle = angle_center - np.pi / 4 + (np.pi / 2) / (args.num_weight_candidates - 1) * i
                weight = np.array([np.cos(angle), np.sin(angle)])

                # 过滤条件1：权重分量不能为负
                if weight[0] >= -1e-7 and weight[1] >= -1e-7:
                    # 过滤条件2：不与邻居节点的权重重复（保持搜索空间的新颖性）
                    if not any(np.linalg.norm(weight - opt_graph.weights[succ]) < 1e-3
                               for succ in opt_graph.succ[sample.optgraph_id]):
                        test_weights.append(weight / np.sum(weight))

            # 使用双曲预测模型预测每个权重的优化结果
            if test_weights:
                if getattr(args, "selection_method", "prediction-guided") in (
                    "ensemble-uncertainty-guided",
                    "ensemble-uncertainty-extreme-guided",
                    "hv-ensemble-guided",
                ):
                    results = predict_hyperbolic_ensemble(args, opt_graph, sample.optgraph_id, test_weights)
                else:
                    results = predict_hyperbolic(args, opt_graph, sample.optgraph_id, test_weights)
                # 将每个测试权重及其预测结果加入候选列表
                uncertainties = results.get('uncertainty', [0.0 for _ in results['predictions']])
                prediction_std = results.get(
                    'prediction_std',
                    [np.zeros(args.obj_num, dtype=float) for _ in results['predictions']],
                )
                for candidate_index, (w, pred, uncertainty, pred_std) in enumerate(zip(
                    test_weights,
                    results['predictions'],
                    uncertainties,
                    prediction_std,
                )):
                    candidates.append({
                        'sample': sample,  # 策略对象
                        'weight': w,  # 优化权重
                        'prediction': pred  # 预测优化后的目标值
                    })
                    candidates[-1]['uncertainty'] = float(uncertainty)
                    candidates[-1]['prediction_std'] = pred_std
                    ensemble_predictions = results.get('ensemble_predictions')
                    if ensemble_predictions is None:
                        candidates[-1]['ensemble_predictions'] = np.asarray([pred], dtype=float)
                    else:
                        candidates[-1]['ensemble_predictions'] = np.asarray(
                            ensemble_predictions, dtype=float
                        )[:, candidate_index, :]

        # ========== 第二部分：贪心选择最优任务 ==========
        if not candidates:
            fallback_count = min(args.num_tasks, len(self.sample_batch))
            elite_batch = self.sample_batch[:fallback_count]
            w_batch = [deepcopy(opt_graph.weights[sample.optgraph_id]) for sample in elite_batch]
            predicted = [deepcopy(sample.objs) for sample in elite_batch]
            return elite_batch, w_batch, predicted

        virtual_ep = [s.objs for s in ep.sample_batch]  # 虚拟帕累托存档（用于模拟）
        elite_batch, w_batch = [], []  # 选中的策略和权重
        mask = [True] * len(candidates)  # 候选是否可选（已选则禁用）

        elite_batch, w_batch, selected_predictions = [], [], []
        selected_regions = []
        selected_extremes = set()
        selection_count = min(args.num_tasks, len(candidates))
        for selection_step in range(selection_count):  # 每轮选择1个任务
            # 评估所有候选的超体积
            selection_method = getattr(args, "selection_method", "prediction-guided")
            if selection_method == "coverage-contribution-guided":
                best_id = self._select_coverage_contribution_candidate(
                    args,
                    candidates,
                    mask,
                    virtual_ep,
                    selected_regions,
                    selected_extremes,
                    selection_step,
                )
            elif selection_method == "contribution-guided":
                scores = self._contribution_scores(args, candidates, mask, virtual_ep)
                best_id = np.argmax(scores)
            elif selection_method == "uncertainty-contribution-guided":
                scores = self._uncertainty_contribution_scores(args, candidates, mask, virtual_ep)
                best_id = np.argmax(scores)
            elif selection_method == "ensemble-uncertainty-guided":
                scores = self._ensemble_uncertainty_scores(args, candidates, mask, virtual_ep)
                best_id = np.argmax(scores)
            elif selection_method == "ensemble-uncertainty-extreme-guided":
                scores = self._ensemble_uncertainty_scores(args, candidates, mask, virtual_ep)
                best_id = self._select_extreme_or_best(
                    candidates,
                    mask,
                    scores,
                    selection_step,
                    getattr(args, "elite_extreme_protection", True),
                )
            elif selection_method == "hv-ensemble-guided":
                scores = self._hv_ensemble_scores(args, candidates, mask, virtual_ep)
                best_id = self._select_extreme_or_best(
                    candidates,
                    mask,
                    scores,
                    selection_step,
                    getattr(args, "elite_extreme_protection", True),
                )
            else:
                hv = [self.compute_hypervolume(virtual_ep + [c['prediction']])
                      if mask[i] else -np.inf for i, c in enumerate(candidates)]
            # 评估所有候选的稀疏度
                sparsity = [self.compute_sparsity(virtual_ep + [c['prediction']])
                            if mask[i] else 0 for i, c in enumerate(candidates)]
                scores = [h - args.sparsity * s for h, s in zip(hv, sparsity)]
                best_id = np.argmax(scores)

            # 选择综合得分最高的候选（超体积越大越好，稀疏度越小越好）
            # 得分 = 超体积 - α × 稀疏度（α > 0 对稀疏度施加惩罚）
            elite = candidates[best_id]  # 选中的任务

            # 更新虚拟帕累托存档：加入预测点，然后只保留非支配解
            virtual_ep.append(elite['prediction'])
            virtual_ep = np.array(virtual_ep)[get_ep_indices(virtual_ep)].tolist()

            # 存储选中任务（用于后续 MOPG 训练）
            elite_batch.append(elite['sample'])
            new_scalar = deepcopy(elite['weight'])
            w_batch.append(new_scalar)
            selected_predictions.append(deepcopy(elite['prediction']))

            # 禁用已选候选（避免重复选择）
            if selection_method == "coverage-contribution-guided":
                selected_regions.append(
                    self._candidate_regions(
                        candidates,
                        getattr(args, "elite_coverage_bins", getattr(args, "num_tasks", 3)),
                    )[best_id]
                )
            mask[best_id] = False

        # 返回选中的任务和对应的预测目标值
        return elite_batch, w_batch, selected_predictions

    # def random_selection(self, args, scalarization_template):
    #     """
    #     随机任务选择策略（基线方法）
    #     随机选择策略和随机生成权重
    #
    #     Args:
    #         args: 配置参数
    #         scalarization_template: 标量化函数模板
    #     Returns:
    #         elite_batch: 随机选中的策略列表
    #         scalarization_batch: 随机生成的权重列表
    #     """
    #     elite_batch, scalarization_batch = [], []
    #     for _ in range(args.num_tasks):
    #         # 随机选择一个策略
    #         elite_idx = np.random.choice(len(self.sample_batch))
    #         elite_batch.append(self.sample_batch[elite_idx])
    #         # 随机生成权重（满足和为1，所有分量非负）
    #         weights = np.random.uniform(args.min_weight, args.max_weight, args.obj_num)
    #         weights = weights / np.sum(weights)  # 归一化使和为1
    #         scalarization = deepcopy(scalarization_template)
    #         scalarization.update_weights(weights)
    #         scalarization_batch.append(scalarization)
    #     return elite_batch, scalarization_batch
