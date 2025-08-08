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

        if self.is_inverse():
            return self._get_state(), -(self.max_steps * 2), True

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

    def is_inverse(self):
        if len(self.word) < 2:
            return False

        last, prev = self.word[-1], self.word[-2]
        return (last == 1 and prev == 3) or (last == 3 and prev == 1) or (last == 2 and prev == 4) or (last == 4 and prev == 2)

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
        return [a for a in (1, 2, 3, 4) if not self.is_inverse_if(a)]

    def is_inverse_if(self, action):
        """
        Return True if `action` would immediately negate the last action.
        That is, action==1 and prev==3, 3<->1, 2<->4, 4<->2.
        """
        if not self.word:
            return False
        prev = self.word[-1]
        return (
                (action == 1 and prev == 3) or
                (action == 3 and prev == 1) or
                (action == 2 and prev == 4) or
                (action == 4 and prev == 2)
        )

    def render(self):
        return ''.join(self.action_to_letter[a] for a in self.word)