import importlib
import torch, numpy as np, random

from burau_representation.rl.global_data import GlobalData, AlgorithmKind
from burau_representation.rl.env import Env


class GlobalSetup:
    """
    Global-only bootstrap:
      - create GlobalData (hard-coded defaults)
      - seed RNGs
      - pick device
      - build env
      - dispatch to algorithm Setup.run()
    """
    def __init__(self, algo_kind: AlgorithmKind):
        self.gd = GlobalData()  # hard-coded global defaults
        self.algo_kind = algo_kind
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _seed(self) -> None:
        s = self.gd.seed
        torch.manual_seed(s)
        np.random.seed(s)
        random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _build_env(self) -> Env:
        env = Env(
            max_steps=self.gd.env_config.max_steps,
            modulo=self.gd.env_config.modulo
        )

        try:
            env.reset(seed=self.gd.seed)
            env.action_space.seed(self.gd.seed)
        except Exception:
            pass
        return env

    def run(self) -> None:
        self._seed()
        env = self._build_env()

        # Import algorithm-local setup
        module_path = f"burau_representation.rl.algorithms.{self.algo_kind.value}.setup"
        mod = importlib.import_module(module_path)

        if not hasattr(mod, "Setup"):
            raise AttributeError(f"{module_path}.Setup class not found")
        setup = mod.Setup(self.gd, self.device, env)

        if not hasattr(setup, "run"):
            raise AttributeError(f"{module_path}.Setup.run() not found")

        setup.run()
