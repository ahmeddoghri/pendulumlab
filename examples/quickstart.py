from pendulumlab import LinearPolicy, PendulumState, rollout

policy = LinearPolicy(theta_gain=-10.0, omega_gain=-2.0)
score, rewards = rollout(policy, PendulumState(theta=0.4, omega=0.0), steps=80)
print(f"quickstart return={score:.2f} first_reward={rewards[0]:.3f} last_reward={rewards[-1]:.3f}")
