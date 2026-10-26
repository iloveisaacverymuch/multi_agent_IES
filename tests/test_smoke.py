from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.ies_env import env


def test_env_step_smoke():
    environment = env()
    states = environment.reset_train()
    state = np.concatenate(states, axis=0)
    action = np.zeros((3, 5))
    outputs = environment.step(action, state, step=0, flag=0)

    assert [item.shape for item in states] == [(1, 7), (1, 7), (1, 7)]
    assert outputs[0].shape == (1, 7)
    assert outputs[4].shape == (2,)


if __name__ == '__main__':
    test_env_step_smoke()
    print('smoke test passed')
