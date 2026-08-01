# pendulumlab

A pendulum that refuses to stay up, and a controller that learns to hold it there without a physics engine the size of a small country.

![CI](https://github.com/ahmeddoghri/pendulumlab/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Most robot learning demos want you to install a simulator before you see a
single reward curve, which is a strange way to teach anyone how control
actually works. pendulumlab skips the download. It is a nonlinear pendulum, a
linear torque policy, and a cross-entropy method loop that tunes the gains
from rollouts, all short enough to read before your coffee goes cold.

This is not trying to out-realism MuJoCo or Isaac Lab. Those are for when you
need the pendulum to also have friction, sensor noise, and a PhD thesis
attached. This is for when you want to actually see the control loop instead
of trusting that it exists somewhere inside a config file.

## Run it

```bash
git clone https://github.com/ahmeddoghri/pendulumlab
cd pendulumlab
pip install -e ".[dev]"
python -m pendulumlab.benchmark
```

This writes `artifacts/reward_curve.csv` and `artifacts/reward_curve.svg`,
which is more receipts than most control demos ever hand you.

## Verified benchmark

Generated locally with `python -m pendulumlab.benchmark`:

```text
controller        avg_return
zero_torque          -54.30
hand_tuned_pd        137.39
cem_policy           217.38
reward_gain           271.68
final_window_avg       1.000
```

Do nothing and the pendulum face-plants at a return of -54.30. A hand-tuned PD
controller gets it to 137.39, which is the control-theory equivalent of
riding a bike with training wheels. CEM, learning entirely from its own
rollouts with nobody hand-turning a dial, lands at 217.38 and keeps the
pendulum upright for almost the entire final stretch.

**Update:** that comparison is against one specific hand-picked PD
guess, not against what the same two-parameter gain family can actually
do. A deterministic 224-point grid search over that exact family (no
learning algorithm, just enumeration, and fewer total evaluations than
CEM's run) reaches 217.70, at or above CEM's number. The 137.39 baseline
wasn't a fair fight; it was an under-tuned guess. `python -m
pendulumlab.benchmark_v2` runs the honest comparison. Details below.

## Does CEM beat a fair baseline, or just a weak one?

`hand_tuned_pd` and `cem_policy` are the exact same policy class:
`LinearPolicy(theta_gain, omega_gain)`, a two-number linear gain. CEM's
job is to search that two-parameter space for a good point in it. So
"CEM (217.38) beats hand-tuned PD (137.39)" answers a narrower question
than it sounds like: did CEM beat one specific guess, not did CEM find
something a plain search over the same space couldn't.

It doesn't. `pendulumlab/grid_search.py` enumerates a fixed 14x16 grid
over `(theta_gain, omega_gain)`, no optimization algorithm at all, and
evaluates each point exactly once:

```bash
python -m pendulumlab.benchmark_v2
```
```
controller        avg_return  policy_evals
zero_torque          -54.30             -
hand_tuned_pd        137.39             1
grid_search_pd       217.70           224
cem_policy            217.38          1008

cem_policy vs grid_search_pd: -0.32 (grid_search ahead or tied)
```

224 grid evaluations, fewer than CEM's 1008 (14 generations x 72
population), reach a score at or above the published `cem_policy` result.
Widening the grid further shows why: the reward landscape over this
two-parameter family is a broad plateau above roughly 217 across a wide
swath of gains, not a narrow peak that needs a smart search to locate.
Four very different gain pairs (`(-30, -6)`, `(-24, -5.5)`, `(-20, -5)`,
`(-28, -6)`) all land within 2 points of each other near the top.

None of this is a code bug: `sim.py`, `cem.py`, and `benchmark.py` all do
exactly what they say, and their published numbers still reproduce
exactly. It's a baseline-strength problem. Report CEM's win against a
weak, hand-picked comparison point and the headline reads like CEM
learned something clever; report it against a fair, deterministic
enumeration of the same tiny policy family, and CEM's edge over honest
search is approximately zero.

## Research trail

A small, dependency-light take on where robot learning actually is right now:

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
