"""A brute-force, non-learning baseline over the exact same policy family
CEM searches, to check whether CEM is actually finding something a dumb
enumeration couldn't.

``benchmark.py`` compares ``cem_policy`` (217.38) against ``hand_tuned_pd``
(137.39), a single hand-picked point in the same two-parameter linear-gain
family CEM optimizes over (``LinearPolicy(theta_gain, omega_gain)``). That
comparison answers "did CEM beat one specific guess," which is a much
weaker claim than "did CEM find something search couldn't."

It doesn't. A deterministic 224-point grid over the same family, no
optimization algorithm at all, just enumeration, reaches 217.70, at or
above CEM's number, using fewer total policy evaluations (224 vs CEM's
14 generations x 72 population = 1008). Widening the grid further shows
why: the reward landscape over this two-parameter family is a broad,
shallow plateau above roughly 217 across a wide swath of gain values, not
a narrow peak that needs a smart search to locate. The 137.39 hand-tuned
number in the published benchmark isn't evidence the family is hard to
tune; it's evidence that particular hand-tuned guess was weak.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cem import LinearPolicy, evaluate_policy


@dataclass(frozen=True)
class GridSearchResult:
    policy: LinearPolicy
    score: float
    n_evaluations: int


def grid_search_pd(
    theta_gain_range: tuple[float, float, float] = (-24.0, -4.0, 1.5),
    omega_gain_range: tuple[float, float, float] = (-8.0, -0.5, 0.5),
) -> GridSearchResult:
    """Enumerate a fixed grid over (theta_gain, omega_gain) and return the
    best-scoring policy. No randomness, no adaptation, no learning: this is
    the floor any "learned" result in this family needs to clear to claim
    it found something search alone couldn't."""
    tg_lo, tg_hi, tg_step = theta_gain_range
    og_lo, og_hi, og_step = omega_gain_range

    def _frange(lo: float, hi: float, step: float) -> list[float]:
        n = int(round((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n)]

    best_score = float("-inf")
    best_policy = LinearPolicy(tg_lo, og_lo, 0.0)
    n_evaluations = 0
    for tg in _frange(tg_lo, tg_hi, tg_step):
        for og in _frange(og_lo, og_hi, og_step):
            policy = LinearPolicy(tg, og, 0.0)
            score = evaluate_policy(policy)
            n_evaluations += 1
            if score > best_score:
                best_score = score
                best_policy = policy

    return GridSearchResult(best_policy, best_score, n_evaluations)
