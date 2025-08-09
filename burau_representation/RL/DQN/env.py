from burau_representation.Classes.Generators import Generators
from burau_representation.Classes.LaurentMatrix import LaurentMatrix


class BurauEnv:
    def __init__(self, max_steps, modulo):
        self.max_steps = max_steps
        self.modulo     = modulo
        self.gens = Generators(modulo)

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

        self.reset()

    def reset(self):
        self.word    = []
        self.turn    = 0
        self.product = LaurentMatrix.identity(self.modulo)

        return []

    def step(self, action):
        if self.turn >= self.max_steps:
            raise ValueError("Episode has ended")

        self.word.append(action)
        self.turn += 1
        self.product = self.product * self.gens[self.action_to_letter[action]]

        '''if self.is_inverse():
            return self._get_state(), -(self.max_steps * 2), True'''

        if self.is_identity():
            return self._get_state(), self.max_steps * 2, True

        done = self.turn >= self.max_steps

        return self._get_state(), 0.0, done

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

    '''def is_inverse(self):
        if len(self.word) < 2:
            return False

        prev, last = self.word[-2], self.word[-1]
        return self.inverse_of.get(prev) == last'''

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