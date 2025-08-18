from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Any, Dict
import torch


class ModelKind(str, Enum):
    LSTM = 'lstm'
    MLP  = 'mlp'


@dataclass(frozen=True)
class Data:
    # Algorithm hyperparameters (edit here to change a run)
    model: ModelKind = ModelKind.LSTM
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'input_size': 1,
        'hidden_dim': 128,
        'num_layers': 1,
        'output_dim': 4,
        'bidirectional': False,
    })

    # Core training loop
    gamma: float = 0.99 # 0.995
    batch_size: int = 64
    num_episodes: int = 32_000

    # Optimizer
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    # Target updates
    target_update: str = 'hard'                 # "hard" | "polyak"
    target_update_freq: Optional[int] = 1_000   # used if "hard"
    tau: Optional[float] = None                 # used if "polyak"

    # Exploration
    epsilon_start: float = 1.0
    target_full_greedy_episodes: int = 75
    target_min_epsilon_episode: int = 25_000

    # Replay
    replay_capacity: int = 100_000

    # Evaluation cadence
    greedy_every_episodes: int = 1_000


# Trainer-relevant knobs only (no env metadata, no capacities)
@dataclass(frozen=True)
class TrainParams:
    num_episodes: int
    batch_size: int
    gamma: float
    # target update policy
    target_update: str # "hard" | "polyak"
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
    env: Any  # Env
    max_steps: int
    policy_net: torch.nn.Module
    target_net: torch.nn.Module
    optimizer: torch.optim.Optimizer
    replay_buffer: Any  # ReplayBuffer
    train_params: TrainParams
    paths: Paths