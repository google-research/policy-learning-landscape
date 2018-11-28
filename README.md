# Policy Learning Landscape

This repository contains code to explore the policy optimiaztion landscape.

## Quick setup

To run cartpole simply do:

```
python3 run_eager_policy_optimization.py --env CartPole-v0 --policy_type discrete
```

To run something from Mujoco you _must_ have it installed and the associated license. To run Hopper-v1 use:

```
python3 run_eager_policy_optimization.py --env Hopper-v1 --policy_type normal --std 0.5
```

Parameters will be saved into `./parameters` as numpy files. After obtaining
some parameters from different runs use the following commands to analyze the landscape.

1. First install eager_pg: `pip install -e .`.

2. Random Pertubations Experiment:
```
cd interpolation_experiments
python paired_random_directions_experiment.py --p1 ./path/to/parameter/1/npy \
--save_dir ./path/to/save/in/ \
--alpha 0.5 --std 0.5 --n_directions 500
```
3. Linear Interpolation Experiment:
```
cd interpolation_experiments
python simple_1d_interpolation_experiment.py --p1 ./path/to/parameter/1/npy \
--p2 ./path/to/parameter/2/npy --save_dir ./path/to/save/in/ \
--stds 5.0 --alpha_start -0.5 --alpha_end 1.5 --n_alphas 2 \
--save_dir ./path/to/save/in
```

Note that interpolation tools only work with continuous policies.


## Code organization

- `eager_pg`: contains a small library to enable quick research in policy
gradient reinforcement learning.
- `analysis_tools`: contains tooling to make nice figures in papers.
- `interpolation_experiments`: Experiments to explore the landscape in policy optimization.

## Disclaimer

This is not an official Google product.
