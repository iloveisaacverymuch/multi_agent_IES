from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedActor(nn.Module):
    def __init__(self, args, obs_dim, action_dim):
        super().__init__()
        self.preference_dim = args.obj_num
        self.register_buffer(
            "default_preference",
            torch.ones(self.preference_dim, dtype=torch.float32) / self.preference_dim,
        )
        input_dim = obs_dim + self.preference_dim
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.mean = nn.Linear(args.hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * args.masac_init_log_std)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.mean.bias, 0)

    def set_default_preference(self, preference):
        preference = torch.tensor(
            preference,
            dtype=torch.float32,
            device=self.default_preference.device,
        ).reshape(-1)
        if preference.numel() != self.preference_dim:
            raise ValueError(
                f"Expected preference dim {self.preference_dim}, got {preference.numel()}."
            )
        total = torch.clamp(preference.sum(), min=1e-8)
        self.default_preference.copy_(preference / total)

    def _prepare_preference(self, state, preference):
        if preference is None:
            preference = self.default_preference
        if not isinstance(preference, torch.Tensor):
            preference = torch.tensor(preference, dtype=torch.float32, device=state.device)
        else:
            preference = preference.to(device=state.device, dtype=torch.float32)

        if preference.dim() == 1:
            if state.dim() == 1:
                return preference
            return preference.unsqueeze(0).expand(state.shape[0], -1)
        return preference

    def forward(self, state, preference=None, evaluation=False):
        preference = self._prepare_preference(state, preference)
        x = torch.cat([state, preference], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std, min=-2.0, max=2.0)
        std = log_std.exp()

        if evaluation:
            action = torch.tanh(mean)
            return action, mean, std, None

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, mean, std, log_prob


class ConditionedMASACCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.state_dim + sum(args.action_dim_per_agent) + args.obj_num
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

    def forward(self, state, joint_action, preference):
        x = torch.cat([state, joint_action, preference], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class ConditionedMASACReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.preference_dim = args.obj_num
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent
        self.capacity = getattr(args, "conditioned_replay_size", args.masac_replay_size)
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
        self.reward_objs = np.zeros((self.capacity, self.N, self.preference_dim), dtype=np.float32)
        self.next_states = [
            np.zeros((self.capacity, self.obs_dim_per_agent[i]), dtype=np.float32)
            for i in range(self.N)
        ]
        self.preferences = np.zeros((self.capacity, self.preference_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.reward_relabel = getattr(args, "conditioned_reward_relabel", True)
        self.preference_grid_size = max(2, getattr(args, "conditioned_preference_grid_size", 11))
        self.endpoint_sample_prob = float(getattr(args, "conditioned_endpoint_sample_prob", 0.4))
        self.grid_sample_prob = float(getattr(args, "conditioned_grid_sample_prob", 0.4))
        self.dirichlet_alpha = float(getattr(args, "conditioned_dirichlet_alpha", 0.7))
        self.reward_normalize = getattr(args, "conditioned_reward_normalize", True)
        self.reward_norm_eps = float(getattr(args, "conditioned_reward_norm_eps", 1e-6))
        self.train_reward_transform = getattr(args, "train_reward_transform", "none")

    def store(self, states, actions, rewards, next_states, done, preference, reward_objs=None):
        for i in range(self.N):
            self.states[i][self.ptr] = np.asarray(states[i], dtype=np.float32).reshape(-1)
            self.actions[i][self.ptr] = np.asarray(actions[i], dtype=np.float32).reshape(-1)
            self.next_states[i][self.ptr] = np.asarray(next_states[i], dtype=np.float32).reshape(-1)
        preference = np.asarray(preference, dtype=np.float32).reshape(-1)
        preference = preference / max(float(preference.sum()), 1e-8)
        self.rewards[self.ptr] = np.asarray(rewards, dtype=np.float32)
        if reward_objs is None:
            self.reward_objs[self.ptr] = 0.0
        else:
            self.reward_objs[self.ptr] = np.asarray(reward_objs, dtype=np.float32).reshape(
                self.N, self.preference_dim
            )
        self.preferences[self.ptr] = preference
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def _sample_preferences(self, batch_size):
        if not self.reward_relabel:
            return None
        if self.preference_dim == 2:
            preferences = np.zeros((batch_size, 2), dtype=np.float32)
            grid = np.linspace(0.0, 1.0, self.preference_grid_size, dtype=np.float32)
            endpoint_prob = np.clip(self.endpoint_sample_prob, 0.0, 1.0)
            grid_prob = np.clip(self.grid_sample_prob, 0.0, 1.0 - endpoint_prob)
            for i in range(batch_size):
                r = np.random.rand()
                if r < endpoint_prob:
                    cost_w = 1.0 if np.random.rand() < 0.5 else 0.0
                elif r < endpoint_prob + grid_prob:
                    cost_w = float(np.random.choice(grid))
                else:
                    cost_w = float(
                        np.random.beta(self.dirichlet_alpha, self.dirichlet_alpha)
                    )
                preferences[i] = np.array([cost_w, 1.0 - cost_w], dtype=np.float32)
            return preferences

        alpha = np.ones(self.preference_dim, dtype=np.float32) * self.dirichlet_alpha
        return np.random.dirichlet(alpha, size=batch_size).astype(np.float32)

    def sample(self, batch_size, device=None, preference_override=None):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty Conditioned MASAC replay buffer.")
        batch_size = min(batch_size, self.size)
        idx = np.random.randint(0, self.size, size=batch_size)
        if preference_override is not None:
            preference_override = np.asarray(preference_override, dtype=np.float32).reshape(-1)
            preference_override = preference_override / max(float(preference_override.sum()), 1e-8)
            preferences_np = np.repeat(preference_override[None, :], batch_size, axis=0)
        else:
            preferences_np = self._sample_preferences(batch_size)
        if preferences_np is None:
            preferences_np = self.preferences[idx]
        reward_objs_np = self.reward_objs[idx]
        if np.any(np.abs(reward_objs_np) > 0):
            reward_input_np = reward_objs_np
            if self.train_reward_transform == "log1p":
                reward_input_np = np.log1p(np.maximum(reward_input_np, 0.0))
            elif self.train_reward_transform != "none":
                raise ValueError(f"Unknown train_reward_transform: {self.train_reward_transform}")
            if self.reward_normalize and self.train_reward_transform == "none":
                scale = np.mean(np.abs(reward_input_np), axis=(0, 1), keepdims=True)
                scale = np.maximum(scale, self.reward_norm_eps)
                reward_input_np = reward_input_np / scale
            rewards_np = -np.einsum("bno,bo->bn", reward_input_np, preferences_np)
        else:
            rewards_np = self.rewards[idx]
        return {
            "states": [torch.tensor(s[idx], dtype=torch.float32, device=device) for s in self.states],
            "actions": [torch.tensor(a[idx], dtype=torch.float32, device=device) for a in self.actions],
            "rewards": torch.tensor(rewards_np, dtype=torch.float32, device=device),
            "next_states": [
                torch.tensor(s[idx], dtype=torch.float32, device=device)
                for s in self.next_states
            ],
            "preferences": torch.tensor(preferences_np, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device).squeeze(-1),
        }


class ConditionedMASAC:
    def __init__(self, args):
        self.device = self._resolve_device(getattr(args, "device", "auto"))
        self.N = args.N
        self.state_dim = args.state_dim
        self.preference_dim = args.obj_num
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
        elif getattr(args, "conditioned_reward_normalize", True):
            self.reward_scale = getattr(args, "conditioned_normalized_reward_scale", 1.0)
        else:
            self.reward_scale = args.masac_reward_scale
        self.policy_delay = args.masac_policy_delay
        self.update_step = 0
        self.use_grad_clip = args.use_grad_clip
        self.actor_grad_clip = args.actor_grad_clip
        self.critic_grad_clip = args.critic_grad_clip
        self.automatic_entropy_tuning = args.masac_auto_alpha
        self.target_entropy = torch.tensor(
            [-float(dim) for dim in self.action_dim_per_agent],
            dtype=torch.float32,
            device=self.device,
        )

        self.actor_dict = {
            agent_id: ConditionedActor(
                args,
                self.obs_dim_per_agent[agent_id],
                self.action_dim_per_agent[agent_id],
            )
            for agent_id in range(self.N)
        }
        self.critic = self._build_critic_dict(args)
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.N, device=self.device))
        self._move_modules_to_device()
        self._build_optimizers()

    @staticmethod
    def _resolve_device(device_arg):
        if device_arg in (None, "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)

    def _move_modules_to_device(self):
        for actor in self.actor_dict.values():
            actor.to(self.device)
        for critic in self.critic.values():
            critic.to(self.device)
        self.target_entropy = self.target_entropy.to(self.device)
        self.log_alpha.data = self.log_alpha.data.to(self.device)
        if self.log_alpha.grad is not None:
            self.log_alpha.grad = self.log_alpha.grad.to(self.device)

    def _move_optimizer_state_to_device(self):
        optimizers = []
        optimizers.extend(getattr(self, "actor_optimizers", []))
        for name in ("critic_optimizer", "alpha_optimizer"):
            opt = getattr(self, name, None)
            if opt is not None:
                optimizers.append(opt)
        for opt in optimizers:
            for state in opt.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)

    def to_device(self, device):
        self.device = self._resolve_device(device)
        self._move_modules_to_device()
        self._move_optimizer_state_to_device()
        return self

    def _build_critic_dict(self, args):
        q1 = ConditionedMASACCritic(args)
        q2 = ConditionedMASACCritic(args)
        target_q1 = deepcopy(q1)
        target_q2 = deepcopy(q2)
        for target in (target_q1, target_q2):
            for param in target.parameters():
                param.requires_grad = False
        return {"q1": q1, "q2": q2, "target_q1": target_q1, "target_q2": target_q2}

    def _build_optimizers(self):
        self._move_modules_to_device()
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
        self._move_modules_to_device()
        self._build_optimizers()

    def set_default_preference(self, preference):
        for actor in self.actor_dict.values():
            if hasattr(actor, "set_default_preference"):
                actor.set_default_preference(preference)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _global_state(self, obs_batch):
        return torch.cat(obs_batch, dim=-1)

    def _joint_action_and_log_prob(self, obs_batch, preference):
        actions = []
        log_probs = []
        for i in range(self.N):
            action, _, _, log_prob = self.actor_dict[i](
                obs_batch[i], preference=preference, evaluation=False
            )
            actions.append(action)
            log_probs.append(log_prob)
        return torch.cat(actions, dim=-1), torch.stack(log_probs, dim=-1)

    def _joint_action_for_actor_update(self, obs_batch, preference, train_agent_id):
        actions = []
        log_prob_i = None
        for i in range(self.N):
            if i == train_agent_id:
                action, _, _, log_prob_i = self.actor_dict[i](
                    obs_batch[i], preference=preference, evaluation=False
                )
            else:
                with torch.no_grad():
                    action, _, _, _ = self.actor_dict[i](
                        obs_batch[i], preference=preference, evaluation=False
                    )
            actions.append(action)
        return torch.cat(actions, dim=-1), log_prob_i

    def update(self, replay_buffer, updates=1, preference_override=None):
        if len(replay_buffer) == 0:
            return
        for _ in range(updates):
            self.update_step += 1
            batch = replay_buffer.sample(
                self.batch_size,
                device=self.device,
                preference_override=preference_override,
            )
            states = batch["states"]
            actions = batch["actions"]
            next_states = batch["next_states"]
            preferences = batch["preferences"]
            rewards = batch["rewards"] * self.reward_scale
            dones = batch["dones"].unsqueeze(-1)

            global_states = self._global_state(states)
            global_next_states = self._global_state(next_states)
            joint_actions = torch.cat(actions, dim=-1)

            with torch.no_grad():
                next_joint_actions, next_log_probs = self._joint_action_and_log_prob(
                    next_states, preferences
                )
                target_q1 = self.critic["target_q1"](
                    global_next_states, next_joint_actions, preferences
                )
                target_q2 = self.critic["target_q2"](
                    global_next_states, next_joint_actions, preferences
                )
                target_q = torch.min(target_q1, target_q2)
                entropy_term = self.alpha.detach().unsqueeze(0) * next_log_probs
                target = rewards + self.gamma * (1.0 - dones) * (target_q - entropy_term)

            q1 = self.critic["q1"](global_states, joint_actions, preferences)
            q2 = self.critic["q2"](global_states, joint_actions, preferences)
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
                        states, preferences, agent_id
                    )
                    q1_pi = self.critic["q1"](
                        global_states, policy_joint_actions, preferences
                    )[:, agent_id]
                    q2_pi = self.critic["q2"](
                        global_states, policy_joint_actions, preferences
                    )[:, agent_id]
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
