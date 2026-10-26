from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.ies_env import env
from src.emomarl.arguments import get_parser
from src.emomarl.warm_up import initialize_weights


def main():
    environment = env()
    states = environment.reset_train()
    state = np.concatenate(states, axis=0)
    action = np.zeros((3, 5))
    outputs = environment.step(action, state, step=0, flag=0)

    args = get_parser().parse_args(['--num_tasks', '1'])
    weights = initialize_weights(args)

    print('state_shapes:', [item.shape for item in states])
    print('step_done:', outputs[3])
    print('warmup_weights:', weights.tolist())


if __name__ == '__main__':
    main()
