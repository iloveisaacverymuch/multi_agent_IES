import numpy as np
import torch
import torch.nn as nn
from src.emomarl.networks import Actor, Critic


class MAPPO:
    """多智能体PPO算法主类（全程 Tensor 计算）"""
    def __init__(self, args):
        """初始化算法参数和网络"""
        # ========== 环境参数 ==========
        self.N = args.N
        self.state_dim = args.state_dim
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent

        # ========== 训练超参数 ==========
        self.batch_size = args.batch_size
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.ppo_epoch = args.ppo_epoch
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.entropy_coef = args.entropy_coef
        self.use_reward_norm = args.use_reward_norm
        self.use_grad_clip = args.use_grad_clip
        self.use_adv_norm = args.use_adv_norm
        self.use_value_clip = args.use_value_clip
        self.value_clip_range = args.value_clip_range
        self.actor_grad_clip = args.actor_grad_clip
        self.critic_grad_clip = args.critic_grad_clip

        # ========== 初始化网络 ==========
        self.actor_dict = {}
        for agent_id in range(self.N):
            actor_input_dim = self.obs_dim_per_agent[agent_id]
            action_dim = self.action_dim_per_agent[agent_id]
            self.actor_dict[agent_id] = Actor(args, actor_input_dim, action_dim)

        self.critic = Critic(args)

        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.actor_lr[id], eps=1e-5)
                                 for id, actor in self.actor_dict.items()]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5)

    def get_value(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        v = self.critic(s)
        return v

    def get_critic_inputs(self, transition_dict, flag):
        key = "states" if flag == "s" else "next_states"
        obs_list = []
        for i in range(self.N):
            val = transition_dict[i][key]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            obs_list.append(val)
        return torch.cat(obs_list, dim=-1)

    def compute_gae(self, rewards, values, next_values, dones):
        length = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        for agent_id in range(self.N):
            gae = 0.0
            for t in reversed(range(length)):
                delta = rewards[t, agent_id] + self.gamma * next_values[t, agent_id] * (1 - dones[t]) - values[t, agent_id]
                gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
                advantages[t, agent_id] = gae
                returns[t, agent_id] = gae + values[t, agent_id]

        return advantages, returns

    def update(self, transition_dict):
        # print("updating")
        """
        更新策略网络和价值网络
        支持 ppo_epochs 和 mini-batch 划分
        """
        # ========== 1. 准备数据 ==========
        critic_inputs_n_s = self.get_critic_inputs(transition_dict, 'n_s')
        critic_inputs_s = self.get_critic_inputs(transition_dict, 's')

        with torch.no_grad():
            value_n_s = self.get_value(critic_inputs_n_s)
            value_s = self.get_value(critic_inputs_s)

        # ========== 2. 收集奖励和 dones ==========
        rewards = []
        for i in range(self.N):
            r = transition_dict[i]['rewards']
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r, dtype=torch.float32)
            rewards.append(r.squeeze(-1))
        rewards = torch.stack(rewards, dim=1)

        d = transition_dict[0]['dones']
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float32)
        dones = d.squeeze(-1)

        if self.use_reward_norm:
            for i in range(self.N):
                mean = rewards[:, i].mean()
                std = rewards[:, i].std(unbiased=False) + 1e-6
                rewards[:, i] = (rewards[:, i] - mean) / std

        # ========== 3. GAE ==========
        advantages, returns = self.compute_gae(rewards, value_s, value_n_s, dones)

        # 优势归一化
        if self.use_adv_norm:
            advantages = (advantages - advantages.mean(dim=0, keepdim=True)) / (
                advantages.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            )

        # ========== 4. 旧价值 ==========
        old_values = []
        for i in range(self.N):
            v = transition_dict[i]['state_value']
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            old_values.append(v.squeeze(-1))
        old_values = torch.stack(old_values, dim=1)

        # ========== 5. 收集 Actor 数据（提前转换为 tensor）==========
        actor_data = []
        for i in range(self.N):
            states = transition_dict[i]['states']
            actions = transition_dict[i]['actions']
            old_log_prob = transition_dict[i]['a_log_prob']

            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.float32)
            if not isinstance(old_log_prob, torch.Tensor):
                old_log_prob = torch.tensor(old_log_prob, dtype=torch.float32)

            actor_data.append({
                'states': states,
                'actions': actions,
                'old_log_prob': old_log_prob.squeeze(-1)
            })

        # ========== 6. 全局数据 ==========
        global_states = critic_inputs_s
        T = advantages.shape[0]

        # ========== 7. PPO 多轮更新 ==========
        for epoch in range(self.ppo_epoch):
            # 每轮重新打乱
            shuffled_indices = torch.randperm(T)

            # Mini-batch 划分
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                batch_idx = shuffled_indices[start:end]

                # ---------- 7.1 Critic 更新 ----------
                global_states_batch = global_states[batch_idx]
                returns_batch = returns[batch_idx]  # [batch, N]
                old_values_batch = old_values[batch_idx]  # [batch, N]

                current_values = self.get_value(global_states_batch)  # [batch, N]

                if self.use_value_clip and self.value_clip_range > 0:
                    value_pred_clipped = old_values_batch + torch.clamp(
                        current_values - old_values_batch,
                        -self.value_clip_range,
                        self.value_clip_range
                    )
                    loss_unclipped = (current_values - returns_batch) ** 2
                    loss_clipped = (value_pred_clipped - returns_batch) ** 2
                    value_loss = torch.max(loss_clipped, loss_unclipped).mean()
                else:
                    value_loss = torch.mean((current_values - returns_batch) ** 2)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
                self.critic_optimizer.step()

                # ---------- 7.2 Actor 更新 ----------
                for i in range(self.N):
                    states_batch = actor_data[i]['states'][batch_idx]
                    actions_batch = actor_data[i]['actions'][batch_idx]
                    old_log_prob_batch = actor_data[i]['old_log_prob'][batch_idx]

                    adv_i = advantages[batch_idx, i].unsqueeze(-1)  # [batch, 1]

                    _, mean, std, _ = self.actor_dict[i](states_batch)
                    normal = torch.distributions.Normal(mean, std)
                    z = torch.atanh(torch.clamp(actions_batch, -0.999, 0.999))
                    log_prob_new = normal.log_prob(z).sum(dim=-1)
                    log_prob_new = log_prob_new - torch.log(1 - actions_batch.pow(2) + 1e-6).sum(dim=-1)

                    ratio = torch.exp(log_prob_new - old_log_prob_batch).unsqueeze(-1)
                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv_i
                    policy_loss = -torch.min(surr1, surr2).mean()

                    entropy = normal.entropy().sum(-1).mean()
                    actor_loss = policy_loss - self.entropy_coef * entropy

                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.actor_dict[i].parameters(), self.actor_grad_clip)
                    self.actor_optimizers[i].step()
