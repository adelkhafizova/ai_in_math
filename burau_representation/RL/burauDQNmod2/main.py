from env import DQNBurau, BurauEnv
import torch
import random
q_net = DQNBurau()
q_net.load_state_dict(torch.load("dqn_burau.pt"))
env = BurauEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = env.reset()

while not env.done:
    with torch.no_grad():
        q_vals = q_net(torch.tensor(state, dtype=torch.float32).to(device))
        action = torch.argmax(q_vals).item()+1
        state = env.step(action)[0]
        print(env.render())
        print(env.get_reward())
    
"""
for i in range(10000):
    state = env.reset()
    action = 0
    for _ in range(32):
        with torch.no_grad():
            choices = [x for x in range(1, 5) if x != 5-action]
            action = random.choice(choices)
            state = env.step(action)
    print(state)
"""

    
    