import numpy as np

from src.emomarl.mappo import MAPPO
from src.emomarl.conditioned_masac import ConditionedMASAC
from src.emomarl.decomposed_conditioned_masac import DecomposedConditionedMASAC
from src.emomarl.masac import MASAC
from src.emomarl.mopg import evaluation
from src.emomarl.networks import Actor, Critic
from src.emomarl.sample import Sample
from src.utils import generate_w_batch_test


def initialize_weights(args):
    if args.obj_num != 2:
        step_size = 1.0 / max(args.num_tasks - 1, 1)
        weights = generate_w_batch_test(reward_dim=args.obj_num, step_size=step_size)
        return weights[: args.num_tasks]
    if args.num_tasks <= 1:
        return np.array([[0.5, 0.5]])
    first_obj_weights = np.linspace(0.0, 1.0, args.num_tasks)
    return np.column_stack((first_obj_weights, 1.0 - first_obj_weights))


def initialize_mappo_sample(args):
    actor_dict = {}
    for agent_id in range(args.N):
        actor_dict[agent_id] = Actor(
            args,
            args.obs_dim_per_agent[agent_id],
            args.action_dim_per_agent[agent_id],
        )
    critic = Critic(args)
    agent = MAPPO(args)
    return Sample(agent, actor_dict, critic, optgraph_id=-1)


def initialize_masac_sample(args):
    agent = MASAC(args)
    return Sample(agent, agent.actor_dict, agent.critic, optgraph_id=-1)


def initialize_conditioned_masac_sample(args):
    agent = ConditionedMASAC(args)
    return Sample(agent, agent.actor_dict, agent.critic, optgraph_id=-1)


def initialize_decomposed_conditioned_masac_sample(args):
    agent = DecomposedConditionedMASAC(args)
    return Sample(agent, agent.actor_dict, agent.critic, optgraph_id=-1)


def initialize_warm_up_batch(args):
    weight_batch = initialize_weights(args)
    sample_batch = []
    for weight in weight_batch:
        if args.algorithm == "decomposed_conditioned_masac":
            sample = initialize_decomposed_conditioned_masac_sample(args)
        elif args.algorithm == "conditioned_masac":
            sample = initialize_conditioned_masac_sample(args)
        elif args.algorithm == "masac":
            sample = initialize_masac_sample(args)
        else:
            sample = initialize_mappo_sample(args)

        sample.source_weight = weight
        if hasattr(sample.agent, "set_default_preference"):
            sample.agent.set_default_preference(weight)
        objs, _, _, _ = evaluation(sample, args=args)
        sample.objs = objs
        sample_batch.append(sample)

    return sample_batch, weight_batch
