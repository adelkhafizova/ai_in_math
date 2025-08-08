import random
import time

from collections import deque

import torch
import torch.nn.functional as F

from burau_representation.scripts.Utils import largest_power_range
from burau_representation.scripts.plot import plot_word_power_range


def Double_DQN(args):
    # 1) Hyperparameters
    num_episodes = args['num_episodes']
    max_steps = args['max_steps']
    batch_size = args['batch_size']
    gamma = args['gamma']
    target_update_freq = args['target_update_freq']

    epsilon = args['epsilon_start']
    epsilon_min = args['epsilon_min']
    epsilon_decay = args['epsilon_decay']

    # 2) Device, environment, and buffer
    device = args['device']
    env = args['env']
    replay_buffer = args['replay_buffer']

    # 3) Models and optimizer
    policy_net = args['policy_net']
    target_net = args['target_net']
    optimizer = args['optimizer']

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 4) Logging and bookkeeping
    filename_prefix = args['filename_prefix']
    steps_done = 0
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    win_count = lose_count = timeout_count = 0
    identities = []

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        env.reset()

        state = torch.zeros(
            max_steps,
            dtype=torch.float32,
            device=device
        )

        state_length = torch.ones(1, dtype=torch.long)
        total_reward = 0.0
        loss_sum     = 0.0
        loss_count   = 0
        prev_max = largest_power_range(env.product)

        power_ranges = []

        for t in range(max_steps):
            # 1) Select action (ε-greedy)
            if random.random() < epsilon and episode % 1000 != 0:
                action = random.choice(env.legal_actions())
            else:
                with torch.no_grad():
                    qvals = policy_net(state, state_length).squeeze(0)

                action = qvals.argmax().item() + 1

            # 2) Step environment
            next_raw_state, raw_r, done = env.step(action)
            curr_max = largest_power_range(env.product)

            if episode % 1000 == 0:
                power_ranges.append(curr_max)

            # 3) Reward shaping + complexity + step cost
            if raw_r != 0.0:
                if raw_r > 0:
                    identity_word = env.render()
                    if identity_word not in identities:
                        identities.append(identity_word)

                        with open('identity.txt', 'a') as f:
                            f.write(identity_word + '\n')
                            torch.save(policy_net.state_dict(), f'{filename_prefix}_policy_net_weights.pth')
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
            next_state_length = torch.tensor([env.turn], dtype=torch.long)
            next_state = torch.zeros(max_steps, dtype=torch.float32, device=device)
            next_state[: env.turn] = torch.tensor(
                next_raw_state,
                dtype=torch.float32,
                device=device
            )

            replay_buffer.push(state, action - 1, reward, state_length, next_state, next_state_length, done)
            state = next_state
            state_length = next_state_length
            steps_done += 1

            # 5) Sample & train
            if len(replay_buffer) >= batch_size:
                s_b, a_b, r_b, sl_b, ns_b, nsl_b, d_b = replay_buffer.sample(batch_size)

                q_p = policy_net(s_b, sl_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # q_n = target_net(ns_b, nsl_b).max(1)[0]

                    next_actions = policy_net(ns_b, nsl_b).argmax(dim=1, keepdim=True)
                    q_n = target_net(ns_b, nsl_b).gather(1, next_actions).squeeze(1)

                    q_t = r_b + gamma * q_n * (1 - d_b)

                # loss = F.mse_loss(q_p, q_t)
                loss = F.smooth_l1_loss(q_p, q_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
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

        # End-of-episode logging and epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        if episode % 100 == 0:
            final_word = env.render()
            final_range = largest_power_range(env.product)

            avg_r = sum(recent_rewards) / len(recent_rewards)
            avg_l = sum(recent_losses) / len(recent_losses)
            total = win_count + lose_count + timeout_count

            end_time = time.time()

            print(
                f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, Elapsed: {end_time - start_time:.6f} seconds, "
                f"MaxPowerRange={final_range}, Wins={win_count / total * 100:.0f}%, Loses={lose_count / total * 100:.0f}%, "
                f"Timeouts={timeout_count / total * 100:.0f}%, FinalWord={final_word}"
            )

            start_time = time.time()

            win_count = lose_count = timeout_count = 0

            if episode % 1000 == 0:
                plot_word_power_range(final_word, power_ranges, file_name=f'{filename_prefix}_power_range_episode_{episode}.png')

                if episode % 10000 == 0 and episode > 0:
                    torch.save(policy_net.state_dict(), f'{filename_prefix}_policy_net_weights_episode_{episode}.pth')