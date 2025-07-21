def calculate_epsilon_min(
    max_steps,
    target_full_greedy_episodes,
    total_episodes=100
):
    if not (0 <= target_full_greedy_episodes <= total_episodes):
        raise ValueError(
            "target_full_greedy_episodes must be between 0 and total_episodes"
        )

    # desired per-episode probability of zero random actions
    p_no_random = target_full_greedy_episodes / total_episodes

    # solve (1 - ε) = p_no_random ** (1/max_steps)
    epsilon_min = 1.0 - p_no_random ** (1.0 / max_steps)

    # clamp into [0,1]
    return float(max(0.0, min(1.0, epsilon_min)))