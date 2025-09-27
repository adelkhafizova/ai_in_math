from dataclasses import dataclass
from enum import Enum

from burau_representation.scripts.utils import get_outputs_path


class AlgorithmKind(str, Enum):
    DQN = 'dqn'
    R2D2 = 'r2d2'
    # PPO = "ppo"
    # A2C = "a2c"


@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration (common to all algorithms)."""
    max_steps: int
    modulo: int


@dataclass(frozen=True)
class GlobalData:
    """
    Global, algorithm-agnostic configuration.
    All defaults are hard-coded here — edit this file to change a run.
    """
    seed: int = 42
    env_config: EnvConfig = EnvConfig(max_steps=350, modulo=3)
    log_every: int = 1_000
    outputs_root: str = get_outputs_path()