# pendulumlab

An inverted pendulum benchmark that keeps the robotics part real: physics loop,
controller, CEM policy search, tests, and a reward curve you can regenerate.

![CI](https://github.com/ahmeddoghri/pendulumlab/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

The point is not to beat MuJoCo on realism. The point is to make the control
stack inspectable. You get a small nonlinear pendulum, a linear torque policy,
and a cross-entropy method loop that tunes the gains from rollouts. The same
idea sits underneath plenty of model-predictive control and sim-to-real work,
but here it is short enough to read before coffee goes cold.

## Run it

```bash
git clone https://github.com/ahmeddoghri/pendulumlab
cd pendulumlab
pip install -e ".[dev]"
python -m pendulumlab.benchmark
```

The command writes:

```text
artifacts/reward_curve.csv
artifacts/reward_curve.svg
```

## Verified benchmark

These numbers were generated locally with `python -m pendulumlab.benchmark`:

```text
controller        avg_return
zero_torque          -54.30
hand_tuned_pd        137.39
cem_policy           217.38
reward_gain           271.68
final_window_avg       1.000
```

## Research trail

This project is a small, dependency-light take on current robot learning work:

- MuJoCo Playground, 2025: https://arxiv.org/abs/2502.08844
- Isaac Lab, 2025: https://arxiv.org/abs/2511.04831
- Genesis, a generative physics engine, 2024: https://arxiv.org/abs/2412.10385
- Model Predictive Control and Reinforcement Learning, 2024: https://arxiv.org/abs/2406.00592

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
