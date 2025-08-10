import random
import time
import os
import json

from collections import deque

import torch
import torch.nn.functional as F

from burau_representation.scripts.Utils import get_outputs_path, get_date_str
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
    win_count = 0
    identities = []

    main_output_dir = os.path.join(get_outputs_path(), filename_prefix + '_' + get_date_str())
    weights_dir = os.path.join(main_output_dir, 'weights')
    power_ranges_dir = os.path.join(main_output_dir, 'power_ranges')
    identity_file = os.path.join(main_output_dir, 'identity.txt')

    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(power_ranges_dir, exist_ok=True)

    # --- save run config once ---
    try:
        run_cfg = {
            'timestamp': get_date_str(),
            'device': str(device),
            'max_steps': max_steps,
            'modulo': args['modulo'],
            'num_episodes': num_episodes,
            'batch_size': batch_size,
            'gamma': gamma,
            'target_update_freq': target_update_freq,
            'epsilon_start': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'replay_capacity': args['buffer_capacity'],
            'model': type(policy_net).__name__,
            'model_params': sum(p.numel() for p in policy_net.parameters())
        }
        with open(os.path.join(main_output_dir, 'run_config.json'), 'w') as f:
            json.dump(run_cfg, f, indent=2)
    except Exception as e:
        print(f'[warn] failed to write run_config.json: {e}')

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()

        state = torch.as_tensor(obs["seq"], dtype=torch.float32, device=device)
        next_state_buf = torch.empty_like(state)
        state_length = torch.tensor([int(obs["length"])], dtype=torch.long)
        mask = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=device)

        total_reward = 0.0
        loss_sum = 0.0
        loss_count = 0

        power_ranges = []

        for t in range(max_steps):
            if random.random() < epsilon and episode % 1000 != 0:
                # legal = torch.nonzero(mask > 0.5, as_tuple=False).squeeze(1).tolist()
                legal = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()  # 0..3
                action = random.choice(legal)  # keep 0..3
            else:
                with torch.no_grad():
                    qvals = policy_net(state, state_length).squeeze(0)

                    # qvals[mask < 0.5] = -1e9
                    qvals[~mask] = -1e9

                action = qvals.argmax().item()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            if episode % 1000 == 0:
                power_ranges.append(env.current_power_range())

            total_reward += reward

            next_state_buf.copy_(torch.as_tensor(next_obs["seq"], dtype=torch.float32, device=device))
            next_state_length = torch.tensor([int(next_obs["length"])], dtype=torch.long)  # CPU
            next_mask = torch.as_tensor(next_info["action_mask"], dtype=torch.bool, device=device)  # [4]

            # replay_buffer.push(state.clone(), action, reward, state_length, next_state_buf.clone(), next_state_length, terminated, next_mask)
            replay_buffer.push(state.clone(), action, reward, state_length, next_state_buf.clone(), next_state_length, done, next_mask)
            # advance without new allocations
            state.copy_(next_state_buf)
            state_length = next_state_length
            mask = next_mask
            steps_done += 1

            if len(replay_buffer) >= batch_size:
                s_b, a_b, r_b, sl_b, ns_b, nsl_b, t_b, nm_b = replay_buffer.sample(batch_size)

                q_p = policy_net(s_b, sl_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next_pol = policy_net(ns_b, nsl_b).clone()
                    # use stored next_state action masks to invalidate illegal actions
                    q_next_pol.masked_fill_(~nm_b, -1e9)

                    # Double DQN: select on policy, evaluate on target
                    next_actions = q_next_pol.argmax(dim=1, keepdim=True)
                    q_n = target_net(ns_b, nsl_b).gather(1, next_actions).squeeze(1)

                    q_t = r_b + gamma * q_n * (1 - t_b)

                loss = F.smooth_l1_loss(q_p, q_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                loss_sum += loss.item()
                loss_count += 1

            # 6) Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                if terminated and reward > 0:
                    win_count += 1

                    identity_word = env.render()

                    if identity_word not in identities:
                        with open(identity_file, 'a') as f:
                            f.write(identity_word + '\n')
                            file_name = os.path.join(weights_dir, 'policy_net_weights.pth')

                        torch.save(policy_net.state_dict(), file_name)
                        identities.append(identity_word)

                break

        # End-of-episode logging and epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        if episode % 100 == 0:
            final_word = env.render()
            final_range = env.current_power_range()

            avg_r = sum(recent_rewards) / len(recent_rewards)
            avg_l = sum(recent_losses) / len(recent_losses)

            elapsed_time = time.time() - start_time

            print(
                f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, Elapsed: {elapsed_time:.6f} seconds, "
                f"MaxPowerRange={final_range}, Wins={win_count}, FinalWord={final_word}"
            )

            start_time = time.time()

            win_count = 0

            if episode % 1000 == 0:
                file_name = os.path.join(power_ranges_dir, f'power_range_episode_{episode}.png')
                plot_word_power_range(final_word, power_ranges, file_name=file_name)

                if episode % 10000 == 0 and episode > 0:
                    file_name = os.path.join(weights_dir, f'policy_net_weights_episode_{episode}.pth')
                    torch.save(policy_net.state_dict(), file_name)