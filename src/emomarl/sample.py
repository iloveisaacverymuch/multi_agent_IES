from copy import deepcopy
import torch
import torch.optim as optim


class Sample:
    def __init__(
        self,
        agent,
        actor_dict,
        critic,
        objs=None,
        optgraph_id=None,
        source_weight=None,
        source_task_id=None,
        source_generation=None,
        source_update_step=None,
        parent_optgraph_id=None,
        episode_reward=None,
        episode_agent_rewards=None,
        episode_objs=None,
    ):
        self.actor = actor_dict
        self.critic = critic
        self.agent = agent
        self.objs = objs
        self.optgraph_id = optgraph_id
        self.source_weight = deepcopy(source_weight) if source_weight is not None else None
        self.source_task_id = source_task_id
        self.source_generation = source_generation
        self.source_update_step = source_update_step
        self.parent_optgraph_id = parent_optgraph_id
        self.episode_reward = episode_reward
        self.episode_agent_rewards = deepcopy(episode_agent_rewards) if episode_agent_rewards is not None else None
        self.episode_objs = deepcopy(episode_objs) if episode_objs is not None else None
        self.link_policy_agent()

    @classmethod
    def copy_from(cls, sample):
        return cls(
            agent=deepcopy(sample.agent),
            actor_dict=deepcopy(sample.actor),
            critic=deepcopy(sample.critic),
            objs=deepcopy(sample.objs) if sample.objs is not None else None,
            optgraph_id=sample.optgraph_id,
            source_weight=deepcopy(getattr(sample, 'source_weight', None)),
            source_task_id=getattr(sample, 'source_task_id', None),
            source_generation=getattr(sample, 'source_generation', None),
            source_update_step=getattr(sample, 'source_update_step', None),
            parent_optgraph_id=getattr(sample, 'parent_optgraph_id', None),
            episode_reward=getattr(sample, 'episode_reward', None),
            episode_agent_rewards=deepcopy(getattr(sample, 'episode_agent_rewards', None)),
            episode_objs=deepcopy(getattr(sample, 'episode_objs', None)),
        )

    def link_policy_agent(self):
        if hasattr(self.agent, 'link_policy_networks'):
            self.agent.link_policy_networks(self.actor, self.critic)
            return

        num_agents = len(self.actor)

        # 绑定网络
        self.agent.actor_dict = self.actor
        self.agent.critic = self.critic

        # 保存旧状态
        old_actor_states = []
        old_critic_state = None

        if hasattr(self.agent, 'actor_optimizers') and self.agent.actor_optimizers:
            for opt in self.agent.actor_optimizers:
                old_actor_states.append(deepcopy(opt.state_dict()))

        if hasattr(self.agent, 'critic_optimizer') and self.agent.critic_optimizer:
            old_critic_state = deepcopy(self.agent.critic_optimizer.state_dict())

        # 重新创建优化器，并沿用 agent 中的学习率配置

        self.agent.actor_optimizers = []
        actor_lr = getattr(self.agent, 'actor_lr', [3e-4] * num_agents)
        if not isinstance(actor_lr, (list, tuple)):
            actor_lr = [actor_lr] * num_agents
        for i in range(num_agents):
            opt = optim.Adam(self.actor[i].parameters(), lr=actor_lr[i], eps=1e-5)
            self.agent.actor_optimizers.append(opt)

        critic_lr = getattr(self.agent, 'critic_lr', 5e-4)
        self.agent.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        # 恢复状态
        for i, state in enumerate(old_actor_states):
            if i < len(self.agent.actor_optimizers):
                try:
                    self.agent.actor_optimizers[i].load_state_dict(state)
                except:
                    pass

        if old_critic_state:
            try:
                self.agent.critic_optimizer.load_state_dict(old_critic_state)
            except:
                pass
