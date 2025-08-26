from enum import Enum


class ModelKind(str, Enum):
    MLP = "mlp"
    LSTM = "lstm"
    DUELING_LSTM = "dueling_lstm"
    # NOISY_LSTM   = "noisy_lstm"


class ExplorationKind(str, Enum):
    EPSILON = "epsilon"
    # NOISY   = "noisy"


class TargetUpdate(str, Enum):
    HARD   = "hard"
    POLYAK = "polyak"


class DqnVariant(str, Enum):
    VANILLA = "vanilla"
    DOUBLE  = "double"


__all__ = ["ModelKind", "ExplorationKind", "TargetUpdate", "DqnVariant"]