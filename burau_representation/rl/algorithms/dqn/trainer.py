import random
import time
import os
import json

from collections import deque

import torch
import torch.nn.functional as F

from burau_representation.scripts.plot import plot_word_power_range
from burau_representation.scripts.utils import raw
from burau_representation.scripts.constants import NEG_LARGE

from burau_representation.rl.algorithms.dqn.enums import TargetUpdate, DqnVariant


class Trainer:
    def __init__(self, spec):
        # 1) Hyperparameters
        self.num_episodes = spec.train_params.num_episodes
        self.max_steps = spec.max_steps
        self.batch_size = spec.train_params.batch_size
        self.gamma = spec.train_params.gamma
        self.variant = spec.train_params.variant
        self.target_update = spec.train_params.target_update
        self.target_update_freq = spec.train_params.target_update_freq
        self.tau = spec.train_params.tau

        self.epsilon = spec.train_params.epsilon_start
        self.epsilon_min = spec.train_params.epsilon_min
        self.epsilon_decay = spec.train_params.epsilon_decay

        # 2) Device, environment, and buffer
        self.device = spec.device
        self.env = spec.env
        self.replay_buffer = spec.replay_buffer

        # 3) Models and optimizer
        self.policy_net = spec.policy_net
        self.target_net = spec.target_net
        self.optimizer = spec.optimizer

        # 4) Logging and bookkeeping
        self.log_every = spec.train_params.log_every
        self.greedy_every_episodes = spec.train_params.greedy_every_episodes

        self.main_output_dir = spec.paths.output_dir
        self.weights_dir = spec.paths.weights_dir
        self.power_ranges_dir = spec.paths.power_ranges_dir
        self.identity_file = spec.paths.identity_file
        self.episodes_csv = spec.paths.episodes_csv

    def Train(self):
        print('self.epsilon', self.epsilon)

        raw(self.target_net).load_state_dict(raw(self.policy_net).state_dict())
        self.target_net.eval()

        steps_done = 0
        recent_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        win_count = 0
        identities = set()
        total_identities = 0
        first_identity_episode = None

        start_time = time.time()

        for episode in range(1, self.num_episodes + 1):
            obs, info = self.env.reset()

            state = torch.as_tensor(obs["seq"], dtype=torch.float32, device=self.device)
            next_state_buf = torch.empty_like(state)
            state_length = torch.tensor([int(obs["length"])], dtype=torch.long)
            mask = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=self.device)

            total_reward = 0.0
            loss_sum = 0.0
            loss_count = 0

            power_ranges = []

            greedy_episode = episode % self.greedy_every_episodes == 0

            for t in range(self.max_steps):
                if random.random() < self.epsilon and not greedy_episode:
                    legal = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
                    action = random.choice(legal)
                else:
                    with torch.no_grad():
                        qvals = self.policy_net(state, state_length).squeeze(0)
                        qvals[~mask] = NEG_LARGE

                    action = qvals.argmax().item()

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated

                if greedy_episode:
                    power_ranges.append(self.env.current_power_range())

                total_reward += reward

                next_state_buf.copy_(torch.as_tensor(next_obs["seq"], dtype=torch.float32, device=self.device))
                next_state_length = torch.tensor([int(next_obs["length"])], dtype=torch.long)
                next_mask = torch.as_tensor(next_info["action_mask"], dtype=torch.bool, device=self.device)  # [4]

                self.replay_buffer.push(
                    state.clone(),
                    action,
                    reward,
                    state_length,
                    next_state_buf.clone(),
                    next_state_length,
                    done,
                    next_mask
                )

                # Advance without new allocations
                state.copy_(next_state_buf)
                state_length = next_state_length
                mask = next_mask
                steps_done += 1

                if len(self.replay_buffer) >= self.batch_size:
                    s_b, a_b, r_b, sl_b, ns_b, nsl_b, t_b, nm_b = self.replay_buffer.sample(self.batch_size)

                    q_p = self.policy_net(s_b, sl_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        if self.variant == DqnVariant.VANILLA:
                            # Vanilla DQN:
                            # directly max over target on s' (mask invalid actions on the target values)
                            q_next_tgt = self.target_net(ns_b, nsl_b).clone()
                            q_next_tgt.masked_fill_(~nm_b, NEG_LARGE)
                            q_n = q_next_tgt.max(dim=1).values
                        elif self.variant == DqnVariant.DOUBLE:
                            # Double DQN:
                            # 1) select a* with policy on s'
                            q_next_pol = self.policy_net(ns_b, nsl_b).clone()
                            q_next_pol.masked_fill_(~nm_b, NEG_LARGE)  # invalidate illegal actions
                            next_actions = q_next_pol.argmax(dim=1, keepdim=True)

                            # 2) evaluate with target on s', a*
                            q_n = self.target_net(ns_b, nsl_b).gather(1, next_actions).squeeze(1)

                        q_t = r_b + self.gamma * q_n * (1 - t_b)

                    loss = F.smooth_l1_loss(q_p, q_t)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                    self.optimizer.step()
                    # Polyak target update (soft)

                    if self.target_update == TargetUpdate.POLYAK and self.tau is not None:
                        with torch.no_grad():
                            for t_param, p_param in zip(raw(self.target_net).parameters(), raw(self.policy_net).parameters()):
                                t_param.data.lerp_(p_param.data, self.tau)

                    loss_sum += loss.item()
                    loss_count += 1

                # 6) Update target network periodically
                if self.target_update == TargetUpdate.HARD and self.target_update_freq is not None and steps_done % self.target_update_freq == 0:
                    raw(self.target_net).load_state_dict(raw(self.policy_net).state_dict())

                if done:
                    if terminated:
                        win_count += 1
                        total_identities += 1
                        if first_identity_episode is None:
                            first_identity_episode = episode

                        identity_word = self.env.render()

                        if identity_word not in identities:
                            identities.add(identity_word)

                            with open(self.identity_file, 'a') as f:
                                f.write(identity_word + '\n')
                                file_name = os.path.join(self.weights_dir, 'policy_net_weights.pth')

                            torch.save(self.policy_net.state_dict(), file_name)

                    break

            # End-of-episode logging and epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            recent_rewards.append(total_reward)
            recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

            if (episode % self.log_every) == 0:
                try:
                    with open(self.episodes_csv, 'a') as f:
                        f.write(f"{episode},{total_reward:.6f},{(recent_losses[-1] if recent_losses else 0.0):.6f},{self.epsilon:.6f},{1 if (first_identity_episode == episode) else 0},{self.env.turn}\n")
                except Exception:
                    pass

            if episode % 100 == 0:
                final_word = self.env.render()
                final_range = self.env.current_power_range()

                avg_r = sum(recent_rewards) / len(recent_rewards)
                avg_l = sum(recent_losses) / len(recent_losses)

                elapsed_time = time.time() - start_time

                print(
                    f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, AvgLoss={avg_l:.5f}, Elapsed: {elapsed_time:.6f} seconds, "
                    f"MaxPowerRange={final_range}, Wins={win_count}, FinalWord={final_word}"
                )

                start_time = time.time()

                win_count = 0

                if greedy_episode:
                    file_name = os.path.join(self.power_ranges_dir, f'power_range_episode_{episode}.png')
                    plot_word_power_range(final_word, power_ranges, file_name=file_name)

                    if episode % 10_000 == 0:
                        file_name = os.path.join(self.weights_dir, f'policy_net_weights_episode_{episode}.pth')
                        torch.save(self.policy_net.state_dict(), file_name)

        try:
            summary = {
                "episodes_run": episode,
                "first_identity_episode": first_identity_episode,
                "total_identities": total_identities,
            }
            with open(os.path.join(self.main_output_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[warn] failed to write summary.json: {e}")