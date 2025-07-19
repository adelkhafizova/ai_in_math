# from ...experiments.test_sympy.burau_sympy import LaurentMatrix
from burau_representation.scripts.free_scripts import largest_power_range
from burau_representation.Classes.Generators import Generators
from burau_representation.Classes.LaurentMatrix import LaurentMatrix

import numpy as np
# import torch.optim as optim
# import sys
# sys.path.append('../../scripts')
# import burau_enchanced as b
# from free_scripts import largest_power_range
# MODULO = 2
# MAX_LENGTH = 32
'''matrices= {
    "A":b.A.convert_to_modulo(MODULO),
    "B": b.B.convert_to_modulo(MODULO),
    "a": b.a.convert_to_modulo(MODULO),
    "b": b.b.convert_to_modulo(MODULO)
}

Id = b.Id.convert_to_modulo(MODULO)'''

   
class BurauEnv:
    def __init__(self, mod, max_length):
        self.mod = mod
        self.max_length = max_length

        self.identity = LaurentMatrix.identity(mod)

        self.gens = Generators(mod)

        self.symbols = {
            0: "",
            1: "A",
            2: "B",
            3: "b",
            4: "a"
        }

        self.reset()

    def reset(self):
        self.turn = 0
        self.word = np.zeros(self.max_length, dtype=int)  # initialized with all zeroes
        self.matrix = self.identity
        self.power_range = 0
        self.winner = None
        self.done = False

        return self.word.copy()
    
    def step(self, action):
        if self.done:
            raise ValueError("Game has ended. Call reset().")

        # Apply move
        self.word[self.turn] = action
        # self.matrix *= matrices[symbols[action]]
        self.matrix *= self.gens[self.symbols[action]]
        self.turn += 1
        # Check for game over
        
        self.done, self.winner = self.check_game_over()
        reward = self.get_reward()
        
        return self.word.copy(), reward, self.done
    """
    def get_reward(self):
        if self.turn != 1 and (self.word[self.turn-1] + self.word[self.turn-2] == 5):
            return -10000
        range = largest_power_range_word(matrices,self.render())
        return 10/(1+range)
    """
    
    def get_reward(self):
        new_power_range = largest_power_range(self.matrix)
        if self.turn != 1 and (self.word[self.turn-1] + self.word[self.turn-2] == 5):
            self.power_range = new_power_range
            return -10
        
        if self.power_range - new_power_range == 1:
            self.power_range = new_power_range
            return 1
        elif self.power_range-new_power_range == 0:
            self.power_range = new_power_range
            return 0
        else:
            self.power_range = new_power_range
            return -1

    def check_game_over(self):
        if self.turn == self.max_length:
            return True, 0
        return False, None

    def render(self):
        # symbols = {0:"",1:"A",2:"B",3:"b",4:"a"}
        word_repr = ""
        for i in self.word:
            word_repr += self.symbols[i]
        return word_repr