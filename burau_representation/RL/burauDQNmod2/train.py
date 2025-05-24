import torch
import torch.optim as optim
import random
from tqdm import trange
from env import BurauEnv, DQNBurau, ReplayBuffer









BATCH_SIZE = 500
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000  # steps
TARGET_UPDATE_FREQ = 50  # steps
REPLAY_CAPACITY = 100000
LR = 1e-3
#TRACKING_NUMBER = 100 #for statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment and networks
env = BurauEnv()
q_net = DQNBurau().to(device)
target_net = DQNBurau().to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

epsilon = EPSILON_START
step_count = 0
num_episodes = 100000

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(1,5))
    with torch.no_grad():
        q_vals = q_net(torch.tensor(state, dtype=torch.float32).to(device))
        return torch.argmax(q_vals).item()+1


performance_stat = []
for episode in trange(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        step_count += 1

        # Decay epsilon
        epsilon = max(EPSILON_END, EPSILON_START - step_count / EPSILON_DECAY)

        # Training step
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)     
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)-1
            rewards = torch.tensor(rewards).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.uint8).to(device)
            # Q(s, a)
            q_values = q_net(states).gather(1, actions).squeeze()
            
            # Target Q(s’, a’) = r + γ max_a’ Q_target(s’, a’)
            with torch.no_grad():
                next_q = target_net(next_states)
                max_next_q = torch.max(next_q, dim=1).values
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync target network
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())


torch.save(q_net.state_dict(), "dqn_burau_3.pt")

