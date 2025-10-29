import numpy as np
import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import sys

from burau_package.classes.generators import Generators
from burau_package.classes.laurent_matrix import LaurentMatrix
from burau_package.scripts.utils import largest_power_range




class BurauEnv(gymnasium.Env):
    """Gym wrapper around your Burau word‐building game."""
    metadata = {"render.modes": ["human"]}

    def __init__(self,mod = 2, obs = 32, reward_fun = "old"):
        super().__init__()
        self.symbols = {-1: "", 1: "A", 2: "B", 3: "b", 4: "a"}
        self.reward_fun = reward_fun
        self.mod = mod
        self.obs =  obs
        self.identity = LaurentMatrix.identity(self.mod)
        self.gens = Generators(mod)
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
        self.word        = []
        self.matrix      = LaurentMatrix.identity(self.mod)
        self.power_range = largest_power_range(self.matrix)
        self.done        = False

        info = {}   # you can optionally return diagnostics here
        return [0]*self.obs, info

    def step(self, action):
        """
        Gymnasium-style step: returns obs, reward, terminated, truncated, info.
        """
        action = action + 1
        if isinstance(action, np.ndarray):
            action = int(action.squeeze())
            

        # apply action
        
        self.word.append(action)
        sym = self.symbols[action]
        self.matrix = self.matrix * self.gens[sym]
        self.turn += 1

        # recompute range & check identity
        is_identity = np.array_equal(self.matrix, self.identity)

        # episode flags
        terminated = is_identity
        truncated  = (self.turn >= self.obs) and not is_identity




        if self.turn > 1 and (self.word[-1] + self.word[-2] == 5):
            self.done = True
            return self._get_obs(), -100, True, truncated, {}

        """
        if largest_power_range(self.matrix) > 10:
            return self._get_obs(), -1, True, truncated, {}
        """
        
        if terminated:
            return self._get_obs(), 1000, terminated, truncated, {} 
        # reward (you can still call your injected reward_fn)
        if self.reward_fun == "old":
            reward = self._get_reward()
        else:
            reward = self._get_reward_new()


        if self.turn >= self.obs - 3:
            reward = reward
        
        # build outputs
        obs  = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_reward(self):
        """Your original reward logic."""
        new_range = largest_power_range(self.matrix)
        delta = self.power_range - new_range
        self.power_range = new_range
        if delta >= 1:
            return 1
        elif delta == 0:
            return 0
        else:
            return -1
        
    def _get_reward_new(self):
        """Your original reward logic."""
        
        new_range = largest_power_range(self.matrix)
        self.power_range = new_range
        if self.turn < self.obs:
            return 0
        else:
            return (self.obs-self.power_range)**2


    def _get_obs(self):
        # return a copy so the buffer can’t be modified in-place
        return ([0]*(self.obs-len(self.word)) + self.word)[-self.obs:]

    def render(self, mode="human"):
        
        s = "".join(self.symbols[i]  for i in self.word if i != 0)
        if mode == "human":
            print(f"[Turn {self.turn}] word = {s or '(empty)'}")
        else:
            super().render(mode=mode)  # just in case

    def _get_action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if self.turn > 0:
            inv = 5 - self.word[-1]
            mask[inv - 1] = 0
        return mask
    
    def legal_actions(self):
            actions = [0,1,2,3]
            if self.turn > 0:
                actions.remove(4-self.word[-1])
                return actions
            else:
                return actions
    


