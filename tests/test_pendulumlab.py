from pendulumlab import LinearPolicy, PendulumState, evaluate_policy, step, train_cem, wrap_angle


def test_wrap_angle_stays_in_range() -> None:
    assert -3.15 < wrap_angle(9.5) < 3.15


def test_physics_moves_state() -> None:
    state = step(PendulumState(0.4, 0.0), torque=-2.0)
    assert state != PendulumState(0.4, 0.0)


def test_stabilizing_policy_beats_zero_torque() -> None:
    zero = evaluate_policy(LinearPolicy(0.0, 0.0))
    tuned = evaluate_policy(LinearPolicy(-10.0, -2.0))
    assert tuned > zero + 80


def test_cem_curve_improves() -> None:
    result = train_cem(seed=11, generations=5, population=32)
    assert result.curve[-1] >= result.curve[0]
    assert evaluate_policy(result.policy) > evaluate_policy(LinearPolicy(0.0, 0.0))
