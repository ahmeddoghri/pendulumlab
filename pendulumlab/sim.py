from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PendulumState:
    theta: float
    omega: float


def wrap_angle(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def step(
    state: PendulumState,
    torque: float,
    dt: float = 0.05,
    gravity: float = 9.81,
    damping: float = 0.08,
    torque_limit: float = 12.0,
) -> PendulumState:
    torque = max(-torque_limit, min(torque_limit, torque))
    theta = wrap_angle(state.theta)
    omega_dot = gravity * math.sin(theta) + torque - damping * state.omega
    omega = max(-8.0, min(8.0, state.omega + dt * omega_dot))
    return PendulumState(wrap_angle(theta + dt * omega), omega)


def reward(state: PendulumState, torque: float) -> float:
    angle = abs(wrap_angle(state.theta))
    return 1.0 - 1.4 * (angle / math.pi) - 0.04 * state.omega * state.omega - 0.002 * torque * torque


def rollout(
    policy: Callable[[PendulumState], float],
    start: PendulumState,
    steps: int = 220,
) -> tuple[float, list[float]]:
    state = start
    total = 0.0
    rewards: list[float] = []
    for _ in range(steps):
        torque = policy(state)
        score = reward(state, torque)
        rewards.append(score)
        total += score
        state = step(state, torque)
    return total, rewards
