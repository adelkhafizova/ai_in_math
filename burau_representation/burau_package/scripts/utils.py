import math
import os
import torch
from datetime import datetime


def calculate_epsilon_min(max_steps,
                          target_full_greedy_episodes,
                          total_episodes=100,
                          target_min_epsilon_episode=50000):

    if not (0 <= target_full_greedy_episodes <= total_episodes):
        raise ValueError(
            "target_full_greedy_episodes must be between 0 and total_episodes"
        )

    # desired per-episode probability of zero random actions
    p_no_random = target_full_greedy_episodes / total_episodes

    # solve (1 - ε) = p_no_random ** (1/max_steps)
    epsilon_min = 1.0 - p_no_random ** (1.0 / max_steps)

    # clamp into [0,1]
    epsilon_min = float(max(0.0, min(1.0, epsilon_min)))

    # decay rate such that: epsilon = 1.0 * decay^target_min_epsilon_episode = epsilon_min
    decay = math.exp(math.log(epsilon_min) / target_min_epsilon_episode)

    return epsilon_min, decay


def largest_power_range(matrix):
    """
    Compute the largest positive and negative powers of x for all entries in a LaurentMatrix.

    Parameters:
        matrix (LaurentMatrix): The matrix to compute power range for.

    Returns:
        int: max(abs(power))
    """
    m = 0
    for row in matrix.matrix:
        for entry in row:
            m = max(m, max(abs(entry.min_power), abs(entry.min_power + len(entry.coefficients) - 1)))
    return m


def get_outputs_path() -> str:
    return os.path.join('burau_representation', 'outputs')


def get_date_str():
    return datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


def raw(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying non-compiled module if torch.compile was used."""
    return getattr(module, "_orig_mod", module)