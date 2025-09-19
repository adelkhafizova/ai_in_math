import copy
import math
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .qnets import DQNBurau
from .env import BurauEnv
from tqdm import trange
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        s   = torch.tensor(state, dtype=torch.float32,    device=self.device)
        a   = torch.tensor([action], dtype=torch.int64,   device=self.device)
        r   = torch.tensor([reward], dtype=torch.float32,  device=self.device)
        ns  = torch.tensor(next_state, dtype=torch.float32,    device=self.device)
        d   = torch.tensor([done], dtype=torch.float32,    device=self.device)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (s, a, r, ns, d)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.cat(actions),
            torch.cat(rewards),
            torch.stack(next_states),
            torch.cat(dones)
        )

    def __len__(self):
        return len(self.buffer)






def dqn(
        modulo,
        max_steps,
        num_episodes=50000,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        buffer_size=100000,
        target_update_freq=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        epsilon_decay=0.9999       # slower decay for more exploration
):

    # Reproducibility
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = exploration_initial_eps


    policy_net = DQNBurau(max_steps).to(device)
    target_net = DQNBurau(max_steps).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(buffer_size, device)
    steps_done = 0
    env = BurauEnv(modulo, max_steps)

    # For simple logging
    recent_rewards = deque(maxlen=100)
    recent_losses  = deque(maxlen=100)

    avg_loss_history = []
    win_count = lose_count = timeout_count = 0


    for episode in trange(1, num_episodes + 1):
        state, _        = env.reset()
        total_reward = 0.0
        loss_sum     = 0.0
        loss_count   = 0
        eval_info    = ""
        prev_max = env.power_range


        for t in range(max_steps):
            
            # 1) Select action (ε-greedy)
            if random.random() < epsilon:

                action = random.choice(env.legal_actions())
            else:
                with torch.no_grad():
                    inp    = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals = policy_net(inp).squeeze(0)
                action = qvals.argmax().item()

            # 2) Step environment
            
            next_state, reward, term, trunc, info = env.step(action)
            done = np.logical_or(term, trunc)

            total_reward += reward

            # 4) Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_done += 1
            

            # 5) Sample & train
            if len(replay_buffer) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(batch_size)
                q_p = policy_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_n = target_net(ns_b).max(1)[0]
                    q_t = r_b + gamma * q_n * (1 - d_b)
                loss = F.mse_loss(q_p, q_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum   += loss.item()
                loss_count += 1

            # 6) Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())


            if loss_count:
                avg_loss = (loss_sum / loss_count)
                avg_loss_history.append(avg_loss)

            if done:
                break

            

        # Record final word and range after episode
        final_word  = env.word
        final_range = env.power_range

        """
        # Periodic evaluation (greedy policy) every 1000 episodes
        if episode % 100 == 0:
            eval_env = BurauEnv(modulo, max_steps)
            s_eval, _   = eval_env.reset()
            for _ in range(max_steps):
                with torch.no_grad():
                
                    inp_eval = torch.tensor(s_eval, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals_eval = policy_net(inp_eval).squeeze(0)
                    
                a_eval = qvals_eval.argmax().item() 
                s_eval, _, term, trunc, _ = eval_env.step(a_eval)
                if term or trunc:
                    break
            eval_word  = eval_env.word
            eval_range = eval_env.power_range
            eval_info  = f", EvalWord={eval_word}, EvalRange={eval_range}"
        """
        # End-of-episode logging and epsilon decay
        epsilon = max(exploration_final_eps, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        if episode % 1000 == 0:
            avg_r = sum(recent_rewards) / len(recent_rewards)
            avg_l = sum(recent_losses) / len(recent_losses)
            total = win_count + lose_count + timeout_count
            win_pct     = win_count / total * 100 if total else 0
            lose_pct    = lose_count / total * 100 if total else 0
            timeout_pct = timeout_count / total * 100 if total else 0
            env.render()
            print(
                f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, "
                f"FinalWord={final_word}, MaxPowerRange={final_range}, "
                f"Wins={win_pct:.0f}%, Loses={lose_pct:.0f}%, Timeouts={timeout_pct:.0f}%" + eval_info
            )
            win_count = lose_count = timeout_count = 0


