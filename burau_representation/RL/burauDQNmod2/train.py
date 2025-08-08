import torch
import torch.optim as optim
import random
from tqdm import trange

from burau_representation.scripts.plot import plot_epochs_avg_loss

from burau_representation.RL.burauDQNmod2.env import BurauEnv
from burau_representation.RL.burauDQNmod2.model import BurauDQN
from burau_representation.RL.burauDQNmod2.ReplayBuffer import ReplayBuffer


def train(
        mod,
        max_length,
        batch_size=500,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
        target_update_freq=50,
        replay_capacity=100000,
        lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment and networks
    env = BurauEnv(mod, max_length)
    q_net = BurauDQN(max_length).to(device)
    target_net = BurauDQN(max_length).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

epsilon = EPSILON_START
step_count = 0
num_episodes = 100000


def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(1, 5))
    with torch.no_grad():
        q_vals = q_net(torch.tensor(state, dtype=torch.float32).to(device))
        return torch.argmax(q_vals).item() + 1


performance_stat = []

    avg_loss_arr = []

    for episode in trange(num_episodes):
        state = env.reset()
        done = False

        loss_sum = 0

        while not done:
            if random.random() < epsilon:
                action = random.choice(range(1, 5))
            with torch.no_grad():
                q_vals = q_net(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_vals).item() + 1

            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_count += 1

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon_start - step_count / epsilon_decay)

            # Training step
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device) - 1
                rewards = torch.tensor(rewards).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.uint8).to(device)
                # Q(s, a)
                q_values = q_net(states).gather(1, actions).squeeze()

                # Target Q(s’, a’) = r + γ max_a’ Q_target(s’, a’)
                with torch.no_grad():
                    next_q = target_net(next_states)
                    max_next_q = torch.max(next_q, dim=1).values
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                loss = torch.nn.functional.mse_loss(q_values, target_q)

                loss_sum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss_arr.append(loss_sum / env.turn)

        # Sync target network
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

    plot_epochs_avg_loss(avg_loss_arr[1:])

torch.save(q_net.state_dict(), "dqn_burau_4.pt")