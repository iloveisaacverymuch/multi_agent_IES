import numpy as np


def training_objective_vector(obj, args):
    obj = np.asarray(obj, dtype=np.float32)
    transform = getattr(args, "train_reward_transform", "none")
    if transform == "none":
        return obj
    if transform == "log1p":
        return np.log1p(np.maximum(obj, 0.0))
    raise ValueError(f"Unknown train_reward_transform: {transform}")


def scalarized_training_reward(obj, weight, args):
    weight = np.asarray(weight, dtype=np.float32)
    weight = weight / max(float(weight.sum()), 1e-8)
    return -float(np.dot(training_objective_vector(obj, args), weight))
