import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from src.emomarl.sample import Sample  # 假设Sample是一个自定义类，包含目标值(objs)和其他策略信息
from src.utils import get_ep_indices  # 外部函数，用于计算帕累托前沿的索引

'''
Define a external pareto class storing all computed policies on the current pareto front.
'''
class EP:
    def __init__(self):
        # 初始化两个空数组：
        # - obj_batch: 存储每个解的目标值，形状为 [n_samples, n_objectives]
        # - sample_batch: 存储对应的Sample对象（包含策略、状态等）
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

    def index(self, indices, inplace=True):
        """
        根据索引筛选解，可选择就地修改或返回副本
        Args:
            indices: 需要保留的索引列表
            inplace: 是否直接修改类属性（True）或返回副本（False）
        """
        if inplace:
            # 就地操作：直接对obj_batch和sample_batch应用索引
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)],  # 转换为int类型索引
                    [self.obj_batch, self.sample_batch])
        else:
            # 非就地操作：返回深拷贝的筛选结果
            return map(
                lambda batch: deepcopy(batch[np.array(indices, dtype=int)]),  # 深拷贝避免引用问题
                [self.obj_batch, self.sample_batch]
            )

    def update(self, sample_batch):
        """
        更新帕累托前沿：
        1. 添加新样本到sample_batch
        2. 提取目标值并合并到obj_batch
        3. 计算当前帕累托前沿并保留非支配解
        """
        # 1. 深拷贝新样本并添加到sample_batch（避免后续修改影响原始数据）
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))

        # 2. 提取每个样本的目标值(sample.objs)并合并到obj_batch
        for sample in sample_batch:
            if len(self.obj_batch) > 0:
                # 非首次添加：垂直堆叠新目标值
                self.obj_batch = np.vstack([self.obj_batch, sample.objs])
            else:
                # 首次添加：直接创建数组
                self.obj_batch = np.array([sample.objs])

        # 3. 如果当前无目标值，直接返回
        if len(self.obj_batch) == 0:
            return

        # 4. 调用外部函数计算帕累托前沿的索引
        #    (假设get_ep_indices返回非支配解在obj_batch中的索引列表)
        ep_indices = get_ep_indices(self.obj_batch)

        # 5. 根据索引保留非支配解（就地修改）
        self.index(ep_indices)
