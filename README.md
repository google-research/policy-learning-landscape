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

### Optional installation

Optionally you can also `pip install -e .` this repository to make `eager_pg`
globally available.

## Code organization

- `eager_pg`: contains a small library to enable quick research in policy
gradient reinforcement learning.
- `analysis_tools`: contains tooling to make nice figures in papers.

## Disclaimer

This is not an official Google product.
