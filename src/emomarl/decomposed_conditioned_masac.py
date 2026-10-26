from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.emomarl.conditioned_masac import ConditionedMASAC


class DecomposedConditionedMASACCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.N
        self.obs_dim_per_agent = args.obs_dim_per_agent
        self.action_dim_per_agent = args.action_dim_per_agent
        self.preference_dim = args.obj_num
        self.hidden_dim = args.hidden_dim

        self.local_q_nets = nn.ModuleList()
        for agent_id in range(self.N):
            input_dim = (
                self.obs_dim_per_agent[agent_id]
                + self.action_dim_per_agent[agent_id]
                + self.preference_dim
            )
            self.local_q_nets.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                )
            )

        mixer_input_dim = args.state_dim + self.preference_dim
        self.hyper_w = nn.Linear(mixer_input_dim, self.N)
        self.hyper_b = nn.Linear(mixer_input_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for net in self.local_q_nets:
            for module in net:
                if isinstance(module, nn.Linear):
                    gain = 1.0 if module.out_features == 1 else np.sqrt(2)
                    nn.init.orthogonal_(module.weight, gain=gain)
                    nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.hyper_w.weight, gain=0.1)
        nn.init.orthogonal_(self.hyper_b.weight, gain=0.1)
        nn.init.constant_(self.hyper_w.bias, 0)
        nn.init.constant_(self.hyper_b.bias, 0)

    def _prepare_preference(self, state, preference):
        if preference.dim() == 1:
            return preference.unsqueeze(0).expand(state.shape[0], -1)
        return preference

    def _split_batch(self, tensor, dims):
        chunks = []
        start = 0
        for dim in dims:
            chunks.append(tensor[:, start:start + dim])
            start += dim
        return chunks

    def local_q_values(self, state, joint_action, preference):
        preference = self._prepare_preference(state, preference)
        obs_batch = self._split_batch(state, self.obs_dim_per_agent)
        action_batch = self._split_batch(joint_action, self.action_dim_per_agent)
        q_values = []
        for agent_id in range(self.N):
            local_input = torch.cat(
                [obs_batch[agent_id], action_batch[agent_id], preference],
                dim=-1,
            )
            q_values.append(self.local_q_nets[agent_id](local_input))
        return torch.cat(q_values, dim=-1)

    def forward(self, state, joint_action, preference):
        preference = self._prepare_preference(state, preference)
        local_q = self.local_q_values(state, joint_action, preference)
        mixer_input = torch.cat([state, preference], dim=-1)
        weights = F.softplus(self.hyper_w(mixer_input)) + 1e-6
        bias = self.hyper_b(mixer_input).squeeze(-1)
        q_tot = (weights * local_q).sum(dim=-1) + bias
        return q_tot


class DecomposedConditionedMASAC(ConditionedMASAC):
    def _build_critic_dict(self, args):
        q1 = DecomposedConditionedMASACCritic(args)
        q2 = DecomposedConditionedMASACCritic(args)
        target_q1 = deepcopy(q1)
        target_q2 = deepcopy(q2)
        for target in (target_q1, target_q2):
            for param in target.parameters():
                param.requires_grad = False
        return {"q1": q1, "q2": q2, "target_q1": target_q1, "target_q2": target_q2}

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
            team_rewards = (batch["rewards"] * self.reward_scale).sum(dim=-1)
            dones = batch["dones"]

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
                entropy_term = (
                    self.alpha.detach().unsqueeze(0) * next_log_probs
                ).sum(dim=-1)
                target = team_rewards + self.gamma * (1.0 - dones) * (
                    target_q - entropy_term
                )

            q1 = self.critic["q1"](global_states, joint_actions, preferences)
            q2 = self.critic["q2"](global_states, joint_actions, preferences)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic["q1"].parameters())
                    + list(self.critic["q2"].parameters()),
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
                    )
                    q2_pi = self.critic["q2"](
                        global_states, policy_joint_actions, preferences
                    )
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
