import numpy as np
import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import sys

# your existing imports
sys.path.append('../../scripts')
import burau_enchanced as b
from free_scripts import largest_power_range




class BurauEnv(gymnasium.Env):
    """Gym wrapper around your Burau word‐building game."""
    metadata = {"render.modes": ["human"]}

    def __init__(self,mod = 2, obs = 32):
        super().__init__()
        self.symbols = {-1: "", 0: "A", 1: "B", 2: "b", 3: "a"}
        self.mod = mod
        self.obs =  obs
        self.identity = b.Id.convert_to_modulo(self.mod)
        self.matrices = {
            "A": b.A.convert_to_modulo(self.mod),
            "B": b.B.convert_to_modulo(self.mod),
            "a": b.a.convert_to_modulo(self.mod),
            "b": b.b.convert_to_modulo(self.mod),
            }

        # 5 discrete actions: 0 (no‐op) or one of the four generators
        self.action_space = spaces.Discrete(len(self.symbols)-1)

        # observation is the full word array of ints in [0,4]
        # could also use MultiDiscrete, but Box works fine here
        self.observation_space = spaces.Box(
            low=0,
            high=len(self.symbols)-1,
            shape=(self.obs,),
            dtype=np.int32,
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Gymnasium-style reset:
        - seed: optional int
        - options: gymnasium options dict (ignored here)
        Always returns (obs, info_dict).
        """
        # let SB3/Gymnasium drive our RNG if it passes a seed
        if seed is not None:
            self.seed(seed)

        # re‐initialize episode
        self.turn        = 0
        self.word        = np.zeros(self.obs, dtype=np.int32)
        self.matrix      = self.identity
        self.power_range = largest_power_range(self.matrix)
        self.done        = False

        obs = self._get_obs()
        info = {}   # you can optionally return diagnostics here
        return obs, info

    def step(self, action):
        """
        Gymnasium-style step: returns obs, reward, terminated, truncated, info.
        """

        if isinstance(action, np.ndarray):
            action = int(action.squeeze())
            
        # record old potential
        old_range = self.power_range

        # apply action
        self.word[self.turn] = action + 1
        sym = self.symbols[action]
        self.matrix = self.matrix * self.matrices[sym]
        self.turn += 1

        # recompute range & check identity
        new_range = largest_power_range(self.matrix)
        is_identity = np.array_equal(self.matrix, self.identity)

        # episode flags
        terminated = is_identity
        truncated  = (self.turn >= self.obs) and not is_identity

        # reward (you can still call your injected reward_fn)
        reward = self._get_reward()

        # update stored range
        self.power_range = new_range

        # build outputs
        obs  = self._get_obs()
        info = {}     # optionally put debug info here

        return obs, reward, terminated, truncated, info

    def _get_reward(self):
        """Your original reward logic."""
        new_range = largest_power_range(self.matrix)
        # avoid cancelling inverses
        if self.turn > 1 and (self.word[self.turn-1] + self.word[self.turn-2] == 5):
            self.power_range = new_range
            return -10
        
        delta = self.power_range - new_range
        self.power_range = new_range
        if self.turn > 10:
            if delta >= 1:
                return 1
            elif delta == 0:
                return 0.1
            else:
                return -1
        else:
            return 0

    def _get_obs(self):
        # return a copy so the buffer can’t be modified in-place
        return self.word.copy()

    def render(self, mode="human"):
        
        s = "".join(self.symbols[i-1]  for i in self.word if i != 0)
        if mode == "human":
            print(f"[Turn {self.turn}] word = {s or '(empty)'}")
        else:
            super().render(mode=mode)  # just in case


