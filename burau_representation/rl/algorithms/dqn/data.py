from pathlib import Path

import torch

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict

from burau_representation.rl.algorithms.dqn.enums import *


@dataclass(frozen=True)
class Data:
    # Algorithm hyperparameters (edit here to change a run)
    
    variant: DqnVariant = DqnVariant.DOUBLE
    model: ModelKind = ModelKind.MLP
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'input_dim': 34,
        'hidden1': 128,
        'hidden2': 128,
        'output_dim': 4,
    })


    # Core training loop
    gamma: float = 0.99
    batch_size: int = 64
    num_episodes: int = 30_100

    # Optimizer
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    # Target updates
    target_update: TargetUpdate = TargetUpdate.HARD
    target_update_freq: Optional[int] = 1_000   # used if "hard"
    tau: Optional[float] = None                 # used if "polyak"

    # Exploration
    epsilon_start: float = 1.0
    target_full_greedy_episodes: int = 50
    target_min_epsilon_episode: int = 30_000

    # Replay
    replay_capacity: int = 100_000

    # Evaluation cadence
    greedy_every_episodes: int = 1_000

    # ---------- Resume options ----------
    # Path to .pth; can be:
    #  - raw model state_dict, or
    #  - checkpoint dict with keys like {'policy_state_dict', 'optimizer_state_dict', 'epsilon', 'epsilon_steps_done', ...}
    load_model_path: Optional[str] = r"D:\Python Projects\ai_in_math\burau_representation\outputs\dqn_lstm_mod3_len350_07_09_2025_01_05_46\weights\policy_net_weights_episode_10000.pth"
    # If True and checkpoint has optimizer state, load it.
    load_optimizer_state: bool = False
    # To reconstruct ε when resuming. If both provided, epsilon_resume_value takes precedence.
    epsilon_resume_steps: Optional[int] = 10000
    epsilon_resume_value: Optional[float] = None


# Trainer-relevant knobs only (no env metadata, no capacities)
@dataclass(frozen=True)
class TrainParams:
    num_episodes: int
    batch_size: int
    gamma: float
    variant: DqnVariant
    # target update policy
    target_update: TargetUpdate
    target_update_freq: Optional[int]
    tau: Optional[float]
    # exploration
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    # logging cadence
    log_every: int
    # evaluation cadence
    greedy_every_episodes: int


# Paths the trainer writes to; prepared by Setup
@dataclass(frozen=True)
class Paths:
    output_dir: str
    weights_dir: str
    power_ranges_dir: str
    identity_file: str
    episodes_csv: str
    run_config_json: str


# Single payload to Trainer.Train(spec)
@dataclass(frozen=True)
class Spec:
    device: torch.device
    env: Any
    max_steps: int
    policy_net: torch.nn.Module
    target_net: torch.nn.Module
    optimizer: torch.optim.Optimizer
    replay_buffer: Any
    train_params: TrainParams
    paths: Paths