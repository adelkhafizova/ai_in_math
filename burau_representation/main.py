import torch
import numpy as np
import random

from burau_representation.RL.DQN.Trains.Double_DQN import Double_DQN

from burau_representation.RL.DQN.Models.LSTM import LSTM
from burau_representation.RL.DQN.ReplayBuffer.ReplayBuffer import ReplayBuffer
from burau_representation.RL.DQN.env import BurauEnv

from burau_representation.scripts.Utils import calculate_epsilon_min

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)

    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ─── Configuration ───────────────────────────────────────────────────────────
    # Environment
    max_steps = 34
    modulo = 2

    # Training hyperparameters
    num_episodes = 2_000_000
    batch_size = 64
    gamma = 0.99
    target_update_freq = 1_000

    # Exploration
    epsilon_start = 1.0
    epsilon_min, epsilon_decay = calculate_epsilon_min(
        max_steps=max_steps,
        target_full_greedy_episodes=75,
        target_min_epsilon_episode=25000
    )

    print('epsilon_min', epsilon_min)
    print('epsilon_decay', epsilon_decay)

    # Replay buffer
    buffer_capacity = 100_000

    # ─── Runtime setup ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Environment & buffer
    env = BurauEnv(max_steps, modulo)
    replay_buffer = ReplayBuffer(buffer_capacity, device, max_steps)

    # ─── Model & optimizer ────────────────────────────────────────────────────────
    policy_net = LSTM(input_size=1).to(device)
    target_net = LSTM(input_size=1).to(device)

    lr = 3e-4
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # ─── Logging / filenames ─────────────────────────────────────────────────────
    filename_prefix = f'lstm_mod{modulo}_length{max_steps}'

    # ─── Pack arguments ───────────────────────────────────────────────────────────
    args = {
        # Environment hyperparam
        'max_steps': max_steps,

        # Training hyperparams
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'gamma': gamma,
        'target_update_freq': target_update_freq,

        # Exploration schedule
        'epsilon_start': epsilon_start,
        'epsilon_min': epsilon_min,
        'epsilon_decay': epsilon_decay,

        # Runtime
        'device': device,
        'env': env,
        'replay_buffer': replay_buffer,
        # 'n_step': n_step,
        # 'nstep_buffer': nstep_buffer,

        # Models & optimizer
        'policy_net': policy_net,
        'target_net': target_net,
        'optimizer': optimizer,

        # Misc
        'filename_prefix': filename_prefix,
    }

    # ─── Launch training ─────────────────────────────────────────────────────────
    Double_DQN(args)
