from copy import deepcopy
import numpy as np

'''
OptGraph is a data structure to store the optimization history.
优化图是一种存储优化历史的数据结构
The optimization history is a rooted forest, and is organized in a tree structure.
优化历史是一种树结构
'''
class OptGraph:
    """
    用于存储优化历史的数据结构，以树形结构组织优化路径（森林结构）。
    记录每个节点的权重、目标值、相对父节点的改进量，并维护父子关系。
    """
    def __init__(self):
        self.weights = []      # 存储每个节点的权重向量（标准化后）
        self.objs = []         # 存储每个节点的目标值向量
        self.delta_objs = []   # 存储当前节点相对于父节点的目标值改进量（Δf）
        self.prev = []         # 存储父节点索引（-1表示根节点）
        self.succ = []         # 存储子节点索引列表（邻接表）

    def insert(self, weights, objs, prev):
        """
        插入一个新节点到优化图中，并更新树结构。
        Args:
            weights: 当前节点的标量化权重向量（形状 [obj_num]）
            objs: 当前节点的目标值向量（形状 [obj_num]）
            prev: 父节点索引（-1表示无父节点）
        Returns:
            当前节点的索引（int）
        """
        self.weights.append(deepcopy(weights))

        # 2. 记录目标值和父节点
        self.objs.append(deepcopy(objs))
        self.prev.append(prev)

        # 3. 计算目标值改进量（Δf）
        if prev == -1:
            # 根节点的改进量为零向量
            self.delta_objs.append(np.zeros_like(objs))
        else:
            # 非根节点：当前目标值 - 父节点目标值
            self.delta_objs.append(objs - self.objs[prev])

        # 4. 更新父子关系
        if prev != -1:
            # 将当前节点添加到父节点的子节点列表中
            self.succ[prev].append(len(self.objs) - 1)

        # 5. 初始化当前节点的子节点列表（新节点默认为叶子节点）
        self.succ.append([])

        # 返回当前节点的索引
        return len(self.objs) - 1