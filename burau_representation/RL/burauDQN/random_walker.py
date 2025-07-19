from env import BurauEnv
from tqdm import trange

import random
import sys

sys.path.append('../../scripts')
import burau_enchanced as b

num_episodes = 100000

env = BurauEnv()

MODULO = 2
MAX_LENGTH = 32

Id = b.Id.convert_to_modulo(MODULO)


def select_action(state):
    valid_actions = [a for a in range(1, 5) if a + state[0] != 5]

    return random.choice(valid_actions)


for episode in trange(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = select_action(state)
        state, _, done = env.step(action)

    print(env.render())