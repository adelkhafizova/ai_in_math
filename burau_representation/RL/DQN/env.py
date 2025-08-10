import numpy as np

import gymnasium as gym
from gymnasium import spaces

from burau_representation.Classes.Generators import Generators
from burau_representation.Classes.LaurentMatrix import LaurentMatrix
from burau_representation.scripts.Utils import largest_power_range


class BurauEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
            self,
            max_steps: int,
            modulo: int,
            *,
            render_mode: str | None = None
    ):
        super().__init__()

        self.max_steps = max_steps
        self.modulo     = modulo
        self.gens = Generators(modulo)
        self.render_mode = render_mode

        self.action_to_letter = {
                1: 'A',
                2: 'B',
                3: 'a',
                4: 'b'
        }

        self.inverse_of = {
            1: 3,
            3: 1,
            2: 4,
            4: 2
        }

        # ---- Gym spaces ----
        # Observation = dict(seq (padded ints 0..4, 0 means pad), length)
        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=0, high=4, shape=(self.max_steps,), dtype=np.int64),
                "length": spaces.Box(low=0, high=self.max_steps, shape=(), dtype=np.int32)
            }
        )

        self.action_space = spaces.Discrete(4)

        # internal state
        self.word = []
        self.turn = 0
        self.product = LaurentMatrix.identity(self.modulo)
        self._current_power_range = 0
        self._seq = np.zeros(self.max_steps, dtype=np.int64)  # 0-padded action history

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.word = []
        self.turn = 0
        self.product = LaurentMatrix.identity(self.modulo)
        self._seq.fill(0)
        self._current_power_range = largest_power_range(self.product)

        obs = {
            'seq': self._seq.copy(),
            'length': np.int32(1)
        }

        info = {
            'action_mask': self._action_mask().astype(np.bool_),
            'length': np.int32(1)
        }

        return obs, info

    def step(self, action_idx: int):
        if not self.action_space.contains(action_idx):
            raise ValueError(f"Action index must be in 0..3, got {action_idx}")

        if self.turn >= self.max_steps:
            raise ValueError("Episode has ended")

        action = action_idx + 1
        self.word.append(action)
        self.turn += 1
        self.product = self.product * self.gens[self.action_to_letter[action]]

        # record this (1..4) into the padded sequence BEFORE any return
        self._seq[self.turn - 1] = action

        new_power_range = largest_power_range(self.product)

        # ---- terminal success: identity ----
        if self.is_identity():
            reward = float(2 * self.max_steps)
            obs = {
                'seq': self._seq.copy(),
                'length': np.int32(self.turn)
            }

            info = {
                'action_mask': self._action_mask().astype(np.bool_),
                'length': np.int32(self.turn),
                'termination_reason': 'identity'
            }

            self._current_power_range = new_power_range

            return obs, reward, True, False, info

        # ---- shaping (moved from trainer): compare power range
        if new_power_range < self._current_power_range:
            reward = 1.0
        elif new_power_range > self._current_power_range:
            reward = -1.0
        else:
            reward = 0.0

        self._current_power_range = new_power_range

        truncated = (self.turn >= self.max_steps)

        obs = {
            'seq': self._seq.copy(),
            'length': np.int32(self.turn)
        }

        info = {
            'action_mask': self._action_mask().astype(np.bool_),
            'length': np.int32(self.turn)
        }

        return obs, float(reward), False, truncated, info

    def _get_state(self):
        return self.word.copy()

    def is_identity(self):
        for i in range(3):
            for j in range(3):
                e = self.product.matrix[i,j]

                if i == j and not e.is_one():
                    return False
                if i != j and not e.is_zero():
                    return False

        return True

    def legal_actions(self):
        """
        Return the list of actions (1–4) that are not the inverse
        of the last taken action. On the first step, all actions
        are legal.
        """
        # If no previous action, all are legal
        if not self.word:
            return [1, 2, 3, 4]

        # Otherwise filter out the inverse of the last action
        forbidden = self.inverse_of[self.word[-1]]
        return [a for a in (1, 2, 3, 4) if a != forbidden]

    def is_inverse_if(self, action):
        """
        Return True if action would immediately negate the last action.
        That is, action==1 and prev==3, 3<->1, 2<->4, 4<->2.
        """
        if not self.word:
            return False

        prev = self.word[-1]
        return self.inverse_of.get(prev) == action

    def render(self):
        return ''.join(self.action_to_letter[a] for a in self.word)

    def _action_mask(self) -> np.ndarray:
        """True for legal, False for illegal inverse; shape (4,)."""
        mask = np.ones(4, dtype=np.bool_)

        if self.word:
            inv = self.inverse_of[self.word[-1]]
            mask[inv - 1] = False

        return mask

    def current_power_range(self):
        return self._current_power_range