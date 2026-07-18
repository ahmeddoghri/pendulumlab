from __future__ import annotations

from pathlib import Path

from .cem import STARTS, LinearPolicy, evaluate_policy, train_cem
from .sim import rollout


def _write_curve(curve: list[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "reward_curve.csv"
    csv.write_text(
        "generation,avg_return\n"
        + "\n".join(f"{idx + 1},{value:.3f}" for idx, value in enumerate(curve))
        + "\n",
        encoding="utf-8",
    )

    width, height = 720, 320
    lo, hi = min(curve), max(curve)
    span = max(1.0, hi - lo)
    points = []
    for idx, value in enumerate(curve):
        x = 40 + idx * (width - 80) / max(1, len(curve) - 1)
        y = height - 40 - ((value - lo) / span) * (height - 80)
        points.append(f"{x:.1f},{y:.1f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#050505"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#52d273" stroke-width="4"/>
<text x="40" y="32" fill="#ffffff" font-family="monospace" font-size="18">CEM reward curve</text>
<text x="40" y="{height - 12}" fill="#a0a0a0" font-family="monospace" font-size="13">generation 1 to {len(curve)}</text>
</svg>
"""
    (out_dir / "reward_curve.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    result = train_cem()
    random_policy = LinearPolicy(0.0, 0.0, 0.0)
    hand_tuned = LinearPolicy(-9.0, -1.4, 0.0)
    learned = result.policy

    rows = [
        ("zero_torque", evaluate_policy(random_policy)),
        ("hand_tuned_pd", evaluate_policy(hand_tuned)),
        ("cem_policy", evaluate_policy(learned)),
    ]
    _write_curve(result.curve, Path("artifacts"))

    last_rewards = [rollout(learned, start)[1][-40:] for start in STARTS]
    upright_reward = sum(sum(seq) / len(seq) for seq in last_rewards) / len(last_rewards)

    print("pendulumlab benchmark: inverted pendulum stabilization")
    print("controller        avg_return")
    for label, score in rows:
        print(f"{label:16s} {score:10.2f}")
    print(f"reward_gain       {rows[-1][1] - rows[0][1]:10.2f}")
    print(f"final_window_avg  {upright_reward:10.3f}")
    print("artifacts         artifacts/reward_curve.csv artifacts/reward_curve.svg")


if __name__ == "__main__":
    main()
