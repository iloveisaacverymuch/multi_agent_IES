# Experiment Output Index

This directory contains the retained evidence for the current thesis method:
**Carbon-Dividend CDA V3 + All-task Mean-HV-guided Evolutionary MADDPG**.
The canonical configuration and paths are in `configs/main_method.json`.

## Primary Evidence

- `main_vs_maddpg_10seed/`: paired ten-seed comparison with fixed-weight
  MADDPG. Mean normalized HV is 0.798119 versus 0.792269; the one-sided paired
  Wilcoxon p-value is 0.041992.
- `all_evolving_cross_algorithm/`: equal-budget five-seed comparison with
  Anchor-guided E-MADDPG, Fixed-weight MADDPG, Fixed-weight MAPPO, MASAC, and
  MOPSO. The main method has the best mean HV and IGD in this comparison.
- `main_convergence_10seed/`: convergence, deterministic evaluation, and
  runtime records for the ten main-method seeds.
- `cda_algorithm_factorial/`: five-seed 2x2 study of Standard CDA versus
  Carbon-Dividend CDA V3 and fixed-weight versus all-task evolutionary MADDPG.
  The complete method has the best mean HV, IGD, sparsity, and point count, but
  the factor effects are trends rather than p < 0.05 results.
- `all_evolving_selection_ablation/`: Mean-HV selection versus random
  preference selection. Mean-HV improves IGD and point count on average, but
  the five-seed difference is not statistically significant.
- `vector_replay_ablation/`: vector reward relabelling versus stale scalar
  rewards and clearing replay after preference changes. Vector relabelling is
  best on all four reported means; significance is not established at five
  seeds (sparsity p = 0.0625 against each alternative).
- `current_carbon_price_sensitivity/`: three-seed sensitivity at carbon prices
  0, 0.005, 0.01, 0.02, and 0.05. Price 0.005 has the best mean HV/IGD; the
  default 0.01 gives the best sparsity, most Pareto points, and lowest minimum
  carbon. No tested difference is significant with only three seeds.

## Figures

- `figures/all_evolving_cross_algorithm_{metrics,pareto}.png`: primary
  cross-algorithm result.
- `figures/main_vs_maddpg_10seed.png`: ten-seed paired result.
- `figures/main_vs_maddpg_convergence_10seed.png`: formal convergence curves.
- `figures/cda_algorithm_factorial.png`: current 2x2 method ablation.
- `figures/all_evolving_selection_ablation.png`: selection ablation.
- `figures/vector_replay_ablation.png`: dynamic-preference replay ablation.
- `figures/current_main_carbon_price_sensitivity_{metrics,pareto}.png`: current
  method carbon-price sensitivity.
- `figures/dispatch_results_main_method/`: 108 PNG dispatch plots covering
  three preferences, four typical days, three users, and electricity/heat/cold.
  The maximum plotted balance residual is 4.55e-13.
- `figures/dispatch_behavior_current_vs_maddpg.png`: aggregate dispatch behavior
  relative to Fixed-weight MADDPG.

The retained `v3_dividend_ablation_5seed_*`,
`v3_dividend_ablation_behavior_5seed*`, and `v3_preference_season_5seed_*`
files are supporting mechanism studies from the development of
Carbon-Dividend CDA V3. They are not the primary current-algorithm comparison.

## Interpretation Rules

Normalized HV values are comparable only within the experiment that produced
them because each analysis uses a shared experiment-specific normalization and
reference point. Claims should distinguish statistically significant evidence
from directional trends. Carbon-Dividend CDA V3 is budget balanced and changes
agent incentives; it does not guarantee lower physical system emissions in
every seed or algorithm setting.

Raw runs required to reproduce all retained tables and figures remain under
`runs/`. Duplicate aliases, smoke runs, preliminary V1/V2 outputs, obsolete
carbon-price pilots, superseded three-seed files, and Python caches were removed
on 2026-10-23.
