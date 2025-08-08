import time
import random
from collections import deque
import torch

from burau_representation.scripts.free_scripts import largest_power_range
from burau_representation.scripts.plot import plot_word_power_range


def Rainbow_DQN(args):
    # 1) Hyperparameters
    num_episodes       = args['num_episodes']
    max_steps          = args['max_steps']
    batch_size         = args['batch_size']
    gamma              = args['gamma']
    target_update_freq = args['target_update_freq']

    epsilon            = args['epsilon_start']
    epsilon_min        = args['epsilon_min']
    epsilon_decay      = args['epsilon_decay']

    per_beta_start     = args.get('per_beta_start', 0.4)
    per_beta_frames    = args.get('per_beta_frames', num_episodes)

    # 2) Device, environment, and buffer
    device             = args['device']
    env                = args['env']
    policy_net         = args['policy_net']
    target_net         = args['target_net']
    optimizer          = args['optimizer']
    replay_buffer      = args['replay_buffer']  # PER buffer

    # 3) Initialize target network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 4) Logging
    filename_prefix    = 'rainbow_DQN_' + args['filename_prefix']
    steps_done         = 0
    recent_rewards     = deque(maxlen=100)
    recent_losses      = deque(maxlen=100)
    win_count = lose_count = timeout_count = 0
    identities         = []

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        env.reset()
        state        = torch.zeros(max_steps, dtype=torch.float32, device=device)
        state_len    = torch.ones(1, dtype=torch.long)
        total_reward = 0.0
        loss_sum     = 0.0
        loss_count   = 0
        prev_max     = largest_power_range(env.product)
        power_ranges = []

        # Anneal PER beta
        per_beta = min(1.0, per_beta_start + (episode / per_beta_frames) * (1.0 - per_beta_start))

        for t in range(max_steps):
            # 1) Action selection
            if random.random() < epsilon and episode % 1000 != 0:
                action = random.choice(env.legal_actions())
            else:
                with torch.no_grad():
                    qvals = policy_net(state.unsqueeze(0), state_len).squeeze(0)
                action = qvals.argmax().item() + 1

            # 2) Environment step
            next_raw, raw_r, done = env.step(action)
            curr_max = largest_power_range(env.product)
            if episode % 1000 == 0:
                power_ranges.append(curr_max)

            # Reward shaping
            if raw_r != 0.0:
                reward = raw_r
                if raw_r > 0:
                    word = env.render()
                    if word not in identities:
                        identities.append(word)
                        with open('identity.txt', 'a') as f:
                            f.write(word + '\n')
                        torch.save(policy_net.state_dict(), f"{filename_prefix}_policy_net.pth")
            else:
                reward = 1.0 if curr_max < prev_max else -1.0 if curr_max > prev_max else 0.0
            prev_max = curr_max
            total_reward += reward

            # 3) Prepare next state
            next_state_len = torch.tensor([env.turn])
            next_state     = torch.zeros(max_steps, dtype=torch.float32, device=device)
            next_state[:env.turn] = torch.tensor(next_raw, dtype=torch.float32, device=device)

            # 4) Store single-step transition
            replay_buffer.push(
                state,
                action - 1,
                reward,
                state_len,
                next_state,
                next_state_len,
                done
            )

            # 5) Advance state
            state, state_len = next_state, next_state_len
            steps_done += 1

            # 6) Training step
            if len(replay_buffer) >= batch_size:
                (s_b, a_b, r_b, sl_b, ns_b, nsl_b, d_b), idxs, weights = replay_buffer.sample(batch_size, beta=per_beta)

                # Current Q-values
                q_p = policy_net(s_b, sl_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                # Double-DQN 1-step target
                with torch.no_grad():
                    next_a   = policy_net(ns_b, nsl_b).argmax(1)
                    q_tgt    = target_net(ns_b, nsl_b).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    q_target = r_b + gamma * q_tgt * (1 - d_b)

                # Loss with PER weights
                loss = (weights * (q_p - q_target).pow(2)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update PER priorities
                td_errors = (q_p - q_target).abs().detach().cpu().numpy()
                replay_buffer.update_priorities(idxs, td_errors)

                loss_sum += loss.item()
                loss_count += 1

            # 7) Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # 8) Episode termination
            if done:
                if raw_r > 0:
                    win_count += 1
                elif raw_r < 0:
                    lose_count += 1
                else:
                    timeout_count += 1
                break

        # End of episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        if episode % 100 == 0:
            final_word  = env.render()
            final_range = largest_power_range(env.product)
            avg_r = sum(recent_rewards) / len(recent_rewards)
            avg_l = sum(recent_losses) / len(recent_losses)
            total = win_count + lose_count + timeout_count
            end_time = time.time()

            print(f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, Elapsed: {end_time - start_time:.6f} seconds, "
                  f"MaxPowerRange={final_range}, Wins={win_count/total*100:.0f}%, Loses={lose_count/total*100:.0f}%, "
                  f"Timeouts={timeout_count/total*100:.0f}%, FinalWord={final_word}")

            start_time = time.time()
            win_count = lose_count = timeout_count = 0

            # Plotting
            if episode % 1000 == 0:
                plot_word_power_range(final_word, power_ranges, file_name=f"{filename_prefix}_power_range_{episode}.png")
                if episode % 10000 == 0:
                    torch.save(policy_net.state_dict(), f'{filename_prefix}_policy_net_weights_episode_{episode}.pth')