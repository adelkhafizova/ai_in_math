import copy
import math
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from burau_representation.RL.Dima.model import BurauDQN
from burau_representation.RL.Dima.buffer import ReplayBuffer
from burau_representation.RL.Dima.env import BurauEnv
from burau_representation.scripts.free_scripts import largest_power_range

from burau_representation.scripts.plot import plot_word_power_range, plot_epochs_avg_loss


def train_fc(
        modulo,
        max_steps,
        num_episodes=2000000,
        batch_size=64,
        gamma=0.99,
        lr=3e-4,
        buffer_capacity=100000,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999       # slower decay for more exploration
):

    # Reproducibility
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = epsilon_start

    # Networks and optimizer
    # policy_net = BurauDQN(input_dim=max_steps, hidden_dim=512).to(device)
    # target_net = BurauDQN(input_dim=max_steps, hidden_dim=512).to(device)

    policy_net = BurauDQN(input_dim=max_steps).to(device)
    target_net = BurauDQN(input_dim=max_steps).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    replay_buffer = ReplayBuffer(buffer_capacity, device)
    steps_done = 0
    env = BurauEnv(max_steps, modulo)

    # For simple logging
    recent_rewards = deque(maxlen=100)
    recent_losses  = deque(maxlen=100)

    avg_loss_history = []
    win_count = lose_count = timeout_count = 0

    identities = []

    for episode in range(1, num_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        loss_sum     = 0.0
        loss_count   = 0
        eval_info    = ""
        prev_max = largest_power_range(env.product)

        for t in range(max_steps):
            # 1) Select action (ε-greedy)
            if random.random() < epsilon:
                action = random.choice(env.legal_actions())
            else:
                with torch.no_grad():
                    inp    = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals = policy_net(inp).squeeze(0)
                action = qvals.argmax().item() + 1

            # 2) Step environment
            next_state, raw_r, done = env.step(action)
            curr_max = largest_power_range(env.product)

            # 3) Reward shaping + complexity + step cost
            if raw_r != 0.0:
                if raw_r > 0:
                    identity_word = env.render()
                    if identity_word not in identities:
                        identities.append(identity_word)
                        """
                        with open('identity.txt', 'a') as f:
                            f.write(identity_word + '\n')
                            
                            torch.save(policy_net.state_dict(), "policy_net_weights.pth")
                        """
                reward = raw_r
            else:
                if curr_max < prev_max:
                    reward = 1.0
                elif curr_max > prev_max:
                    reward = -1.0
                else:
                    reward = 0.0

            prev_max = curr_max
            total_reward += reward

            # 4) Store transition
            replay_buffer.push(state, action - 1, reward, next_state, done)
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

            if done:
                if raw_r > 0:
                    win_count += 1
                elif raw_r < 0:
                    lose_count += 1
                else:
                    timeout_count += 1
                break

            if loss_count:
                avg_loss = (loss_sum / loss_count)
                avg_loss_history.append(avg_loss)

        # Record final word and range after episode
        final_word  = env.render()
        final_range = largest_power_range(env.product)

        # Periodic evaluation (greedy policy) every 1000 episodes
        if episode % 1000 == 0:
            eval_env = BurauEnv(max_steps, modulo)
            s_eval   = eval_env.reset()
            eval_ranges = []
            for _ in range(max_steps):
                eval_ranges.append(largest_power_range(eval_env.product))
                with torch.no_grad():
                    inp_eval = torch.tensor(s_eval, dtype=torch.float32, device=device).unsqueeze(0)
                    qvals_eval = policy_net(inp_eval).squeeze(0)
                a_eval = qvals_eval.argmax().item() + 1
                s_eval, _, d_eval = eval_env.step(a_eval)
                if d_eval:
                    break
            eval_word  = eval_env.render()
            eval_range = largest_power_range(eval_env.product)
            eval_info  = f", EvalWord={eval_word}, EvalRange={eval_range}"
            """
            plot_word_power_range(eval_word, eval_ranges, file_name=f'3_power_range_episode_{episode}.png')
            """
        # End-of-episode logging and epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        if episode % 100 == 0:
            avg_r = sum(recent_rewards) / len(recent_rewards)
            avg_l = sum(recent_losses) / len(recent_losses)
            total = win_count + lose_count + timeout_count
            win_pct     = win_count / total * 100 if total else 0
            lose_pct    = lose_count / total * 100 if total else 0
            timeout_pct = timeout_count / total * 100 if total else 0
            print(
                f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, "
                f"FinalWord={final_word}, MaxPowerRange={final_range}, "
                f"Wins={win_pct:.0f}%, Loses={lose_pct:.0f}%, Timeouts={timeout_pct:.0f}%" + eval_info
            )
            win_count = lose_count = timeout_count = 0


