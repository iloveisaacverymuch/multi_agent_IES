import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class Actor(nn.Module):
    """
    策略网络：输入状态，输出动作均值和标准差
    最终动作通过 tanh 缩放到 [-1, 1]
    """

    def __init__(self, args, actor_input_dim, action_dim, init_log_std=0.5):
        super(Actor, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(actor_input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.mean = nn.Linear(args.hidden_dim, action_dim)

        # 对数标准差（可学习参数，独立于状态）
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)

    def forward(self, state, evaluation=False):
        """
        前向传播
        Args:
            state: 状态 [batch, state_dim]
            deterministic: 是否确定性输出（推理时使用）
        Returns:
            action: 动作 [batch, action_dim] 范围 [-1, 1]
            log_prob: 动作的对数概率 [batch]
            mean: 原始均值 [batch, action_dim]
            std: 标准差 [action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)

        # 限制 log_std 范围，防止数值不稳定
        log_std = torch.clamp(self.log_std, min=-2.0, max=2.0)
        std = log_std.exp()

        if evaluation:
            # 确定性输出
            action = torch.tanh(mean)
            return action, mean, std, None

        # 随机采样
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # 重参数化采样 [batch, action_dim]
        action = torch.tanh(z)

        # 计算 log_prob（考虑 tanh 变换的雅可比行列式）
        log_prob_z = normal.log_prob(z).sum(dim=-1)  # [batch]
        # tanh 变换：log_prob = log_prob_z - ∑ log(1 - tanh(z)^2)
        log_prob = log_prob_z - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, mean, std, log_prob

    # def get_entropy(self, mean, std):
    #     """计算策略熵"""
    #     normal = torch.distributions.Normal(mean, std)
    #     entropy = normal.entropy().sum(dim=-1).mean()
    #     return entropy


class Critic(nn.Module):
    """
    价值网络：输入状态，输出状态价值 V(s)
    """

    def __init__(self, args):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.N)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value  # [batch, N]
