"""Tests for the weak-baseline finding: does CEM beat a fair, non-learning
search over the same policy family, not just a weak hand-tuned guess?"""

from __future__ import annotations

from pendulumlab.cem import LinearPolicy, evaluate_policy, train_cem
from pendulumlab.grid_search import grid_search_pd

# --- the finding: the published hand-tuned baseline is weak -----------------

def test_a_trivial_grid_search_matches_or_beats_the_published_cem_result():
    """No learning algorithm, just enumeration over the same 2-parameter
    family cem_policy searches, using fewer total evaluations than CEM's
    published run (14 generations x 72 population = 1008)."""
    result = grid_search_pd()
    assert result.n_evaluations < 1008
    assert result.score >= 217.0  # matches/exceeds the published 217.38


def test_grid_search_beats_the_published_hand_tuned_baseline_by_a_lot():
    hand_tuned_score = evaluate_policy(LinearPolicy(-9.0, -1.4, 0.0))
    result = grid_search_pd()
    assert result.score > hand_tuned_score + 75


def test_cem_does_not_meaningfully_beat_the_fair_baseline():
    """The real comparison: CEM vs. a fair (non-learning) baseline over the
    same family, not CEM vs. one weak hand-picked guess."""
    cem_result = train_cem()
    cem_score = evaluate_policy(cem_result.policy)
    grid_result = grid_search_pd()
    # CEM should not be dramatically ahead of a dumb enumeration; if it
    # were, that would actually support the original "CEM learns something
    # search can't" framing. It isn't.
    assert cem_score <= grid_result.score + 5.0


def test_reward_landscape_is_a_plateau_not_a_narrow_peak():
    """Several very different gain pairs all land within a few points of
    each other near the top of the family -- the family isn't hard to
    optimize, which is why a dumb grid matches a smart search."""
    candidates = [(-30.0, -6.0), (-24.0, -5.5), (-20.0, -5.0), (-28.0, -6.0)]
    scores = [evaluate_policy(LinearPolicy(tg, og, 0.0)) for tg, og in candidates]
    assert max(scores) - min(scores) < 2.0


# --- deterministic and reproducible ------------------------------------------

def test_grid_search_is_deterministic():
    a = grid_search_pd()
    b = grid_search_pd()
    assert a.score == b.score
    assert a.policy == b.policy


# --- the original benchmark is unaffected -----------------------------------

def test_original_cem_module_untouched():
    import pendulumlab.cem as cem_module

    assert not hasattr(cem_module, "grid_search_pd")


def test_original_benchmark_numbers_still_reproduce():
    from pendulumlab.cem import STARTS

    zero = evaluate_policy(LinearPolicy(0.0, 0.0, 0.0))
    hand_tuned = evaluate_policy(LinearPolicy(-9.0, -1.4, 0.0))
    assert round(zero, 2) == -54.30
    assert round(hand_tuned, 2) == 137.39
    assert len(STARTS) == 6
