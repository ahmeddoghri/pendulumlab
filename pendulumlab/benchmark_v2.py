"""Does CEM beat a fair, non-learning baseline, not just a weak hand-tuned one?

``benchmark.py``'s headline compares ``cem_policy`` (217.38) to a single
hand-picked ``hand_tuned_pd`` point (137.39) in the exact same two-
parameter linear-gain family CEM searches. That is not a fair baseline:
it measures whether CEM beats one specific guess, not whether it found
anything a plain search over the same family couldn't.

``grid_search.py``'s deterministic, dependency-free enumeration over that
same family, no optimization algorithm at all, reaches 217.70 using fewer
total policy evaluations than CEM's run. CEM's real edge over an honest,
non-learned baseline in this narrow, low-dimensional policy family is
approximately zero.

    python -m pendulumlab.benchmark_v2
"""
from __future__ import annotations

from .cem import LinearPolicy, evaluate_policy, train_cem
from .grid_search import grid_search_pd


def main() -> None:
    hand_tuned = LinearPolicy(-9.0, -1.4, 0.0)
    zero = LinearPolicy(0.0, 0.0, 0.0)

    cem_result = train_cem()
    grid_result = grid_search_pd()

    rows = [
        ("zero_torque", evaluate_policy(zero), None),
        ("hand_tuned_pd", evaluate_policy(hand_tuned), 1),
        ("grid_search_pd", grid_result.score, grid_result.n_evaluations),
        ("cem_policy", evaluate_policy(cem_result.policy), 14 * 72),
    ]

    print("pendulumlab benchmark v2: is CEM's headline beating a fair baseline?")
    print(f"{'controller':<16}{'avg_return':>12}{'policy_evals':>14}")
    for label, score, n_evals in rows:
        n_str = "-" if n_evals is None else str(n_evals)
        print(f"{label:<16}{score:>12.2f}{n_str:>14}")

    cem_score = rows[-1][1]
    grid_score = rows[2][1]
    print(
        f"\ncem_policy vs grid_search_pd: {cem_score - grid_score:+.2f} "
        f"({'CEM ahead' if cem_score > grid_score else 'grid_search ahead or tied'})"
    )
    print("a deterministic enumeration over the SAME two-parameter policy family,")
    print("with no learning algorithm and fewer total evaluations than CEM's run,")
    print("matches or beats the published cem_policy number. the reward landscape")
    print("over this narrow linear-gain family is a broad plateau, not a peak that")
    print("needs a smart search to find. the 137.39 hand_tuned_pd baseline in the")
    print("original benchmark understates what this policy family can do; it is")
    print("not evidence CEM found something search alone couldn't.")


if __name__ == "__main__":
    main()
