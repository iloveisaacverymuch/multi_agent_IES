import argparse
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_list(value, item_type):
    if isinstance(value, list):
        return [item_type(item) for item in value]
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value.split(",")
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError("Expected a list, for example [5,5,4]")
    return [item_type(item) for item in parsed]


def parse_int_list(value):
    return parse_list(value, int)


def parse_float_list(value):
    return parse_list(value, float)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def get_parser():
    parser = argparse.ArgumentParser(description="Evolutionary multi-agent RL")

    parser.add_argument("--obj_num", type=int, default=2, help="Number of objectives")
    parser.add_argument("--num_tasks", type=int, default=6, help="Tasks per generation")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--total_generation", type=int, default=12, help="Total generations")
    parser.add_argument("--warmup_iter", type=int, default=30, help="Warm-up iterations")
    parser.add_argument("--update_iter", type=int, default=12, help="Evolution update iterations")
    parser.add_argument("--pbuffer_num", type=int, default=100, help="Performance buffers")
    parser.add_argument("--pbuffer_size", type=int, default=2, help="Performance buffer size")
    parser.add_argument("--num_weight_candidates", type=int, default=8, help="Weight candidates")
    parser.add_argument("--sample_iter", type=int, default=2, help="Offspring sampling interval")
    parser.add_argument("--ppo_iter", type=int, default=2, help="Worker result interval")
    parser.add_argument("--sparsity", type=float, default=10, help="Sparsity penalty")
    parser.add_argument(
        "--selection_method",
        type=str,
        default="ensemble-uncertainty-guided",
        choices=[
            "prediction-guided",
            "contribution-guided",
            "coverage-contribution-guided",
            "uncertainty-contribution-guided",
            "ensemble-uncertainty-guided",
            "ensemble-uncertainty-extreme-guided",
            "hv-ensemble-guided",
        ],
        help="Elite task selection method.",
    )
    parser.add_argument(
        "--market_mechanism",
        type=str,
        default="carbon_aware",
        choices=[
            "standard",
            "carbon_aware",
            "carbon_aware_v2",
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ],
        help="Electricity trading mechanism used by the IES environment.",
    )
    parser.add_argument(
        "--cda_carbon_price",
        type=float,
        default=0.02,
        help="Carbon price coefficient for carbon-aware CDA matching.",
    )
    parser.add_argument(
        "--cda_dynamic_carbon_price",
        type=parse_bool,
        default=False,
        help="Use a time-varying carbon price in carbon-aware CDA.",
    )
    parser.add_argument(
        "--cda_carbon_pressure_scale",
        type=float,
        default=1.0,
        help="Sensitivity of dynamic CDA carbon price to carbon and market pressure.",
    )
    parser.add_argument(
        "--cda_carbon_intensity_ref",
        type=float,
        default=1.0,
        help="Reference electric carbon intensity used by dynamic carbon pricing.",
    )
    parser.add_argument(
        "--cda_carbon_price_min_multiplier",
        type=float,
        default=0.5,
        help="Minimum multiplier for dynamic CDA carbon price.",
    )
    parser.add_argument(
        "--cda_carbon_price_max_multiplier",
        type=float,
        default=3.0,
        help="Maximum multiplier for dynamic CDA carbon price.",
    )
    parser.add_argument(
        "--elite_coverage_bins",
        type=int,
        default=3,
        help="Number of cost-axis regions used by coverage-contribution elite selection.",
    )
    parser.add_argument(
        "--elite_extreme_protection",
        type=parse_bool,
        default=True,
        help="Protect cost-priority and carbon-priority tasks in coverage-contribution selection.",
    )
    parser.add_argument(
        "--elite_uncertainty_kappa",
        type=float,
        default=0.20,
        help="Exploration bonus strength for uncertainty-contribution elite selection.",
    )
    parser.add_argument(
        "--elite_uncertainty_reference",
        type=str,
        default="ep_population",
        choices=["ep", "population", "ep_population"],
        help="Reference set used to estimate distance-based prediction uncertainty.",
    )
    parser.add_argument(
        "--elite_ensemble_size",
        type=int,
        default=5,
        help="Number of bootstrap predictors used by ensemble-uncertainty selection.",
    )
    parser.add_argument(
        "--elite_ensemble_sample_ratio",
        type=float,
        default=0.80,
        help="Bootstrap sample ratio for each ensemble predictor.",
    )
    parser.add_argument(
        "--elite_ensemble_max_samples",
        type=int,
        default=32,
        help="Maximum historical samples used by ensemble predictor; 0 keeps all samples.",
    )
    parser.add_argument(
        "--save_dir",
        default=str(PROJECT_ROOT / "outputs" / "runs" / "default"),
        help="Output directory",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="mappo",
        choices=[
            "mappo",
            "fixed_weight_mappo",
            "masac",
            "conditioned_masac",
            "decomposed_conditioned_masac",
        ],
        help="RL algorithm",
    )

    parser.add_argument("--N", type=int, default=3, help="Number of agents")
    parser.add_argument(
        "--action_dim_per_agent",
        type=parse_int_list,
        default=[5, 5, 4],
        help="Action dimensions per agent",
    )
    parser.add_argument(
        "--obs_dim_per_agent",
        type=parse_int_list,
        default=[7, 7, 7],
        help="Observation dimensions per agent",
    )
    parser.add_argument("--state_dim", type=int, default=21, help="Global state dimension")

    parser.add_argument(
        "--actor_lr",
        type=parse_float_list,
        default=[3e-4, 3e-4, 3e-4],
        help="Actor learning rates",
    )
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer_size", type=int, default=1440, help="MAPPO buffer size")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--use_reward_norm", type=parse_bool, default=False)
    parser.add_argument("--use_grad_clip", type=parse_bool, default=True)
    parser.add_argument("--actor_grad_clip", type=float, default=0.5)
    parser.add_argument("--critic_grad_clip", type=float, default=0.5)
    parser.add_argument("--use_adv_norm", type=parse_bool, default=True)
    parser.add_argument("--use_value_clip", type=parse_bool, default=True)
    parser.add_argument("--value_clip_range", type=float, default=0.2)
    parser.add_argument("--ppo_epoch", type=int, default=3)
    parser.add_argument(
        "--train_reward_transform",
        type=str,
        default="none",
        choices=["log1p", "none"],
        help="Transform per-step objective vectors for training rewards only.",
    )

    parser.add_argument("--masac_replay_size", type=int, default=50000)
    parser.add_argument("--masac_start_steps", type=int, default=256)
    parser.add_argument("--masac_updates_per_iter", type=int, default=12)
    parser.add_argument("--masac_update_every", type=int, default=8)
    parser.add_argument("--masac_random_steps", type=int, default=512)
    parser.add_argument("--masac_action_noise", type=float, default=0.15)
    parser.add_argument("--masac_noise_decay_steps", type=int, default=5000)
    parser.add_argument("--masac_noise_min_scale", type=float, default=0.2)
    parser.add_argument("--masac_policy_delay", type=int, default=1)
    parser.add_argument("--masac_diverse_task_selection", type=parse_bool, default=True)
    parser.add_argument("--masac_tau", type=float, default=0.005)
    parser.add_argument("--masac_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--masac_reward_scale", type=float, default=1e-4)
    parser.add_argument("--masac_auto_alpha", type=parse_bool, default=True)
    parser.add_argument("--masac_init_log_std", type=float, default=0.3)

    parser.add_argument("--conditioned_centralized_learner", type=parse_bool, default=True)
    parser.add_argument("--conditioned_shared_learner", type=parse_bool, default=True)
    parser.add_argument("--conditioned_eval_grid_size", type=int, default=11)
    parser.add_argument("--conditioned_replay_size", type=int, default=200000)
    parser.add_argument("--conditioned_reward_relabel", type=parse_bool, default=True)
    parser.add_argument("--conditioned_preference_grid_size", type=int, default=11)
    parser.add_argument("--conditioned_endpoint_sample_prob", type=float, default=0.4)
    parser.add_argument("--conditioned_grid_sample_prob", type=float, default=0.4)
    parser.add_argument("--conditioned_dirichlet_alpha", type=float, default=0.7)
    parser.add_argument("--conditioned_reward_normalize", type=parse_bool, default=True)
    parser.add_argument("--conditioned_reward_norm_eps", type=float, default=1e-6)
    parser.add_argument("--conditioned_normalized_reward_scale", type=float, default=1.0)
    parser.add_argument("--conditioned_specialist_updates", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--use_orthogonal_init", type=parse_bool, default=True)
    parser.add_argument("--use_relu", type=parse_bool, default=True)

    return parser
