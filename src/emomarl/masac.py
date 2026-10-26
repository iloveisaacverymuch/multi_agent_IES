from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.emomarl.networks import Actor


class MASACCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.state_dim + sum(args.action_dim_per_agent)
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.q = nn.Linear(args.hidden_dim, args.N)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.q.bias, 0)

    def forward(self, state, joint_action):
        x = torch.cat([state, joint_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class MASACReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent
        self.capacity = args.masac_replay_size
        self.ptr = 0
        self.size = 0
        self.states = [
            np.zeros((self.capacity, self.obs_dim_per_agent[i]), dtype=np.float32)
            for i in range(self.N)
        ]
        self.actions = [
            np.zeros((self.capacity, self.action_dim_per_agent[i]), dtype=np.float32)
            for i in range(self.N)
        ]
        self.rewards = np.zeros((self.capacity, self.N), dtype=np.float32)
        self.next_states = [
            np.zeros((self.capacity, self.obs_dim_per_agent[i]), dtype=np.float32)
            for i in range(self.N)
        ]
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def store(self, states, actions, rewards, next_states, done):
        for i in range(self.N):
            self.states[i][self.ptr] = np.asarray(states[i], dtype=np.float32).reshape(-1)
            self.actions[i][self.ptr] = np.asarray(actions[i], dtype=np.float32).reshape(-1)
            self.next_states[i][self.ptr] = np.asarray(next_states[i], dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = np.asarray(rewards, dtype=np.float32)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty MASAC replay buffer.")
        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": [torch.tensor(s[idx], dtype=torch.float32) for s in self.states],
            "actions": [torch.tensor(a[idx], dtype=torch.float32) for a in self.actions],
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "next_states": [torch.tensor(s[idx], dtype=torch.float32) for s in self.next_states],
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32).squeeze(-1),
        }


class MASAC:
    def __init__(self, args):
        self.N = args.N
        self.state_dim = args.state_dim
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.alpha_lr = args.masac_alpha_lr
        self.gamma = args.gamma
        self.tau = args.masac_tau
        self.batch_size = args.batch_size
        if getattr(args, "train_reward_transform", "none") == "log1p":
            self.reward_scale = 1.0
        else:
            self.reward_scale = args.masac_reward_scale
        self.policy_delay = args.masac_policy_delay
        self.update_step = 0
        self.use_grad_clip = args.use_grad_clip
        self.actor_grad_clip = args.actor_grad_clip
        self.critic_grad_clip = args.critic_grad_clip
        self.automatic_entropy_tuning = args.masac_auto_alpha
        self.target_entropy = torch.tensor(
            [-float(dim) for dim in self.action_dim_per_agent], dtype=torch.float32
        )

        self.actor_dict = {}
        for agent_id in range(self.N):
            self.actor_dict[agent_id] = Actor(
                args,
                self.obs_dim_per_agent[agent_id],
                self.action_dim_per_agent[agent_id],
                init_log_std=args.masac_init_log_std,
            )

        self.critic = self._build_critic_dict(args)
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.N))
        self._build_optimizers()

    def _build_critic_dict(self, args):
        q1 = MASACCritic(args)
        q2 = MASACCritic(args)
        target_q1 = deepcopy(q1)
        target_q2 = deepcopy(q2)
        for target in (target_q1, target_q2):
            for param in target.parameters():
                param.requires_grad = False
        return {"q1": q1, "q2": q2, "target_q1": target_q1, "target_q2": target_q2}

    def _build_optimizers(self):
        actor_lr = self.actor_lr
        if not isinstance(actor_lr, (list, tuple)):
            actor_lr = [actor_lr] * self.N
        self.actor_optimizers = [
            torch.optim.Adam(self.actor_dict[i].parameters(), lr=actor_lr[i], eps=1e-5)
            for i in range(self.N)
        ]
        critic_params = list(self.critic["q1"].parameters()) + list(self.critic["q2"].parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr, eps=1e-5)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, eps=1e-5)

    def link_policy_networks(self, actor_dict, critic):
        self.actor_dict = actor_dict
        if isinstance(critic, dict) and {"q1", "q2"}.issubset(critic):
            self.critic = critic
            if "target_q1" not in self.critic:
                self.critic["target_q1"] = deepcopy(self.critic["q1"])
            if "target_q2" not in self.critic:
                self.critic["target_q2"] = deepcopy(self.critic["q2"])
            for key in ("target_q1", "target_q2"):
                for param in self.critic[key].parameters():
                    param.requires_grad = False
        self._build_optimizers()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _global_state(self, obs_batch):
        return torch.cat(obs_batch, dim=-1)

    def _joint_action_and_log_prob(self, obs_batch):
        actions = []
        log_probs = []
        for i in range(self.N):
            action, _, _, log_prob = self.actor_dict[i](obs_batch[i], evaluation=False)
            actions.append(action)
            log_probs.append(log_prob)
        return torch.cat(actions, dim=-1), torch.stack(log_probs, dim=-1)

    def _joint_action_for_actor_update(self, obs_batch, train_agent_id):
        actions = []
        log_prob_i = None
        for i in range(self.N):
            if i == train_agent_id:
                action, _, _, log_prob_i = self.actor_dict[i](obs_batch[i], evaluation=False)
            else:
                with torch.no_grad():
                    action, _, _, _ = self.actor_dict[i](obs_batch[i], evaluation=False)
            actions.append(action)
        return torch.cat(actions, dim=-1), log_prob_i

    def update(self, replay_buffer, updates=1):
        if len(replay_buffer) == 0:
            return
        for _ in range(updates):
            self.update_step += 1
            batch = replay_buffer.sample(self.batch_size)
            states = batch["states"]
            actions = batch["actions"]
            next_states = batch["next_states"]
            rewards = batch["rewards"] * self.reward_scale
            dones = batch["dones"].unsqueeze(-1)

            global_states = self._global_state(states)
            global_next_states = self._global_state(next_states)
            joint_actions = torch.cat(actions, dim=-1)

            with torch.no_grad():
                next_joint_actions, next_log_probs = self._joint_action_and_log_prob(next_states)
                target_q1 = self.critic["target_q1"](global_next_states, next_joint_actions)
                target_q2 = self.critic["target_q2"](global_next_states, next_joint_actions)
                target_q = torch.min(target_q1, target_q2)
                entropy_term = self.alpha.detach().unsqueeze(0) * next_log_probs
                target = rewards + self.gamma * (1.0 - dones) * (target_q - entropy_term)

            q1 = self.critic["q1"](global_states, joint_actions)
            q2 = self.critic["q2"](global_states, joint_actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic["q1"].parameters()) + list(self.critic["q2"].parameters()),
                    self.critic_grad_clip,
                )
            self.critic_optimizer.step()

            if self.update_step % self.policy_delay == 0:
                actor_log_probs = []
                for agent_id in range(self.N):
                    policy_joint_actions, log_prob_i = self._joint_action_for_actor_update(
                        states, agent_id
                    )
                    q1_pi = self.critic["q1"](global_states, policy_joint_actions)[:, agent_id]
                    q2_pi = self.critic["q2"](global_states, policy_joint_actions)[:, agent_id]
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (
                        self.alpha.detach()[agent_id] * log_prob_i - q_pi
                    ).mean()

                    self.actor_optimizers[agent_id].zero_grad()
                    actor_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_dict[agent_id].parameters(),
                            self.actor_grad_clip,
                        )
                    self.actor_optimizers[agent_id].step()
                    actor_log_probs.append(log_prob_i.detach())

                if self.automatic_entropy_tuning:
                    log_probs = torch.stack(actor_log_probs, dim=-1)
                    alpha_loss = -(
                        self.log_alpha.unsqueeze(0)
                        * (log_probs + self.target_entropy.unsqueeze(0)).detach()
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()

            self.soft_update_targets()

    def soft_update_targets(self):
        for source_key, target_key in (("q1", "target_q1"), ("q2", "target_q2")):
            for source_param, target_param in zip(
                self.critic[source_key].parameters(),
                self.critic[target_key].parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)
