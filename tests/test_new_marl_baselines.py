from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_strict_algorithm_comparison import load_actors
from scripts.train_ippo_baseline import IPPO, empty_rollout
from scripts.train_maddpg_baseline import ACT_DIMS, OBS_DIMS, ReplayBuffer
from scripts.train_matd3_baseline import MATD3


def matd3_args():
    return SimpleNamespace(
        device="cpu",
        gamma=0.99,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        hidden_dim=16,
        actor_lr=3e-4,
        critic_lr=5e-4,
    )


def ippo_args():
    return SimpleNamespace(
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=2,
        minibatch_size=8,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )


def test_matd3_updates_twin_critics_and_delays_actor():
    torch.manual_seed(7)
    np.random.seed(7)
    agent = MATD3(matd3_args())
    replay = ReplayBuffer(64, sum(OBS_DIMS), OBS_DIMS, ACT_DIMS)
    for index in range(32):
        obs = [np.random.randn(dim).astype(np.float32) for dim in OBS_DIMS]
        next_obs = [np.random.randn(dim).astype(np.float32) for dim in OBS_DIMS]
        actions = [np.random.uniform(-1, 1, dim).astype(np.float32) for dim in ACT_DIMS]
        replay.add(
            np.concatenate(obs),
            obs,
            actions,
            np.random.randn(3).astype(np.float32),
            np.concatenate(next_obs),
            next_obs,
            index == 31,
        )

    before = [parameter.detach().clone() for parameter in agent.actors[0].parameters()]
    agent.update(replay, batch_size=8)
    after_first = [parameter.detach().clone() for parameter in agent.actors[0].parameters()]
    assert all(torch.equal(first, second) for first, second in zip(before, after_first))

    agent.update(replay, batch_size=8)
    after_second = list(agent.actors[0].parameters())
    assert any(not torch.equal(first, second) for first, second in zip(before, after_second))
    modules = [agent.actors, agent.critics_1, agent.critics_2]
    assert all(
        torch.isfinite(parameter).all()
        for module in modules
        for parameter in module.parameters()
    )


def test_ippo_actions_are_bounded_and_update_is_finite():
    torch.manual_seed(11)
    np.random.seed(11)
    agent = IPPO(ippo_args())
    rollout = empty_rollout()
    next_obs = None
    for step in range(16):
        obs = [np.random.randn(dim).astype(np.float32) for dim in OBS_DIMS]
        actions, log_probs, values = agent.act(obs)
        assert all(np.max(np.abs(action)) <= 1.0 for action in actions)
        for agent_id in range(3):
            rollout[agent_id]["obs"].append(obs[agent_id])
            rollout[agent_id]["actions"].append(actions[agent_id])
            rollout[agent_id]["log_probs"].append(log_probs[agent_id])
            rollout[agent_id]["rewards"].append(float(np.random.randn()))
            rollout[agent_id]["dones"].append(float(step == 15))
            rollout[agent_id]["values"].append(values[agent_id])
        next_obs = [np.random.randn(dim).astype(np.float32) for dim in OBS_DIMS]

    agent.update(rollout, next_obs, terminal=True)
    deterministic = agent.act(next_obs, evaluation=True)
    assert all(np.max(np.abs(action)) <= 1.0 for action in deterministic)
    assert all(
        torch.isfinite(parameter).all()
        for module in (agent.actors, agent.values)
        for parameter in module.parameters()
    )


def test_strict_evaluator_loads_matd3_and_ippo_policy_formats(tmp_path):
    matd3 = MATD3(matd3_args())
    matd3_path = tmp_path / "matd3.pt"
    torch.save(
        {
            "format": "matd3_state_dict_v1",
            "hidden_dim": 16,
            "actor_state_dicts": [actor.state_dict() for actor in matd3.actors],
        },
        matd3_path,
    )
    loaded_matd3 = load_actors(matd3_path)
    assert set(loaded_matd3) == {0, 1, 2}

    ippo = IPPO(ippo_args())
    ippo_path = tmp_path / "ippo.pt"
    torch.save(
        {
            "format": "ippo_state_dict_v1",
            "hidden_dim": 16,
            "actor_state_dicts": [actor.state_dict() for actor in ippo.actors],
        },
        ippo_path,
    )
    loaded_ippo = load_actors(ippo_path)
    observation = torch.zeros(OBS_DIMS[0])
    action = loaded_ippo[0](observation, evaluation=True)
    assert action.shape == (ACT_DIMS[0],)
    assert torch.max(torch.abs(action)) <= 1.0
