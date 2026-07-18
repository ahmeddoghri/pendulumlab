from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .sim import PendulumState, rollout, wrap_angle


@dataclass(frozen=True)
class LinearPolicy:
    theta_gain: float
    omega_gain: float
    bias: float = 0.0

    def __call__(self, state: PendulumState) -> float:
        return self.theta_gain * wrap_angle(state.theta) + self.omega_gain * state.omega + self.bias


@dataclass(frozen=True)
class TrainResult:
    policy: LinearPolicy
    curve: list[float]
    generations: int


STARTS = [
    PendulumState(0.42, 0.0),
    PendulumState(-0.42, 0.0),
    PendulumState(0.30, 0.45),
    PendulumState(-0.30, -0.45),
    PendulumState(0.55, -0.35),
    PendulumState(-0.55, 0.35),
]


def evaluate_policy(policy: LinearPolicy, starts: list[PendulumState] | None = None) -> float:
    starts = starts or STARTS
    returns = [rollout(policy, start)[0] for start in starts]
    return sum(returns) / len(returns)


def _mean_std(elites: list[tuple[float, tuple[float, float, float]]]) -> tuple[list[float], list[float]]:
    cols = list(zip(*(weights for _, weights in elites)))
    means = [sum(col) / len(col) for col in cols]
    stds = []
    for idx, col in enumerate(cols):
        var = sum((value - means[idx]) ** 2 for value in col) / len(col)
        stds.append(max(0.08, math.sqrt(var)))
    return means, stds


def train_cem(
    seed: int = 11,
    generations: int = 14,
    population: int = 72,
    elite_frac: float = 0.18,
) -> TrainResult:
    rng = random.Random(seed)
    mean = [-10.0, -2.0, 0.0]
    std = [5.0, 2.0, 0.8]
    elite_n = max(3, int(population * elite_frac))
    best_score = float("-inf")
    best_weights = tuple(mean)
    curve: list[float] = []

    for _ in range(generations):
        scored: list[tuple[float, tuple[float, float, float]]] = []
        scored.append((evaluate_policy(LinearPolicy(*best_weights)), best_weights))
        for _ in range(population - 1):
            weights = tuple(mean[i] + std[i] * rng.gauss(0.0, 1.0) for i in range(3))
            policy = LinearPolicy(*weights)
            score = evaluate_policy(policy)
            scored.append((score, weights))
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored[0][0] > best_score:
            best_score, best_weights = scored[0]
        elites = scored[:elite_n]
        mean, std = _mean_std(elites)
        std = [value * 0.82 for value in std]
        curve.append(best_score)

    return TrainResult(LinearPolicy(*best_weights), curve, generations)
