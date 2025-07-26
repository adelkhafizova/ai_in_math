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
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
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
"""
        if episode % 10000 == 0:
            plot_epochs_avg_loss(avg_loss_history, filename=f'3_avg_loss_up_to_{episode}.png')

    plot_epochs_avg_loss(avg_loss_history, filename='3_avg_loss.png')
"""

def train_fc_early_stop(
        modulo,
        max_steps,
        num_episodes=5000000,
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
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = epsilon_start

    # Networks and optimizer
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

            # 2a) Prune if hopeless (can't reach power=1)
            steps_left = max_steps - (t + 1)
            if curr_max > steps_left:
                reward = -100.0
                timeout_count += 1

                # store prune transition
                replay_buffer.push(state, action - 1, reward, next_state, True)
                state = next_state
                steps_done += 1

                # train on this last transition if possible
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
                    # immediate loss logging
                    avg_loss_history.append(loss_sum / loss_count)

                break

            # 2b) Terminal check with raw_r
            if done:
                # win
                if raw_r > 0:
                    reward = raw_r
                    with open('identity.txt', 'a') as f:
                        f.write(env.render() + '\n')
                    win_count += 1

                # inverse
                elif raw_r < 0:
                    reward = raw_r
                    lose_count += 1

                # timeout without identity
                else:
                    reward = -100.0
                    timeout_count += 1

                # store terminal transition
                replay_buffer.push(state, action - 1, reward, next_state, True)
                state = next_state
                steps_done += 1

                # train on this last transition if possible
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
                    # immediate loss logging
                    avg_loss_history.append(loss_sum / loss_count)

                break

            # 3) Non-terminal reward shaping
            if raw_r != 0.0:
                reward = raw_r
                if raw_r > 0:
                    with open('identity.txt', 'a') as f:
                        f.write(env.render() + '\n')
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
            replay_buffer.push(state, action - 1, reward, next_state, False)
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
                # immediate loss logging
                avg_loss_history.append(loss_sum / loss_count)

            # 6) Update target network periodically
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

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
            plot_word_power_range(eval_word, eval_ranges,
                                  file_name=f'power_range_episode_{episode}.png')

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
                f"[Episode {episode:6d}] AvgReward={avg_r:.3f}, "
                f"AvgLoss={avg_l:.5f}, FinalWord={final_word}, "
                f"MaxPowerRange={final_range}, "
                f"Wins={win_pct:.0f}%, Loses={lose_pct:.0f}%, "
                f"Timeouts={timeout_pct:.0f}%"
                + eval_info
            )
            win_count = lose_count = timeout_count = 0

    plot_epochs_avg_loss(avg_loss_history)


def train_transformer(
        modulo,
        max_steps,
        num_episodes=1_000_000,
        batch_size=64,
        gamma=0.99,
        lr=3e-4,
        buffer_capacity=100_000,
        target_update_freq=1_000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999):

    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = epsilon_start

    # model setup
    policy_net = BurauTransformer(
        seq_len=max_steps, d_model=64, nhead=4, num_layers=2, output_dim=4
    ).to(device)
    target_net = BurauTransformer(
        seq_len=max_steps, d_model=64, nhead=4, num_layers=2, output_dim=4
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    replay_buffer   = ReplayBuffer(buffer_capacity, device)
    env             = BurauEnv(max_steps, modulo)
    steps_done      = 0

    recent_rewards  = deque(maxlen=100)
    recent_losses   = deque(maxlen=100)
    avg_loss_history = []

    win_count = lose_count = timeout_count = 0

    for episode in range(1, num_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        loss_sum     = 0.0
        loss_count   = 0

        # initial potential φ(s₀)
        # prev_phi = -largest_power_range(env.product)
        prev_largest_power_range = largest_power_range(env.product)

        for t in range(max_steps):
            # ε-greedy with illegal-action filtering
            if random.random() < epsilon:
                candidates = env.legal_actions()
                action = random.choice(candidates)
            else:
                with torch.no_grad():
                    inp   = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
                    qvals = policy_net(inp).squeeze(0)
                action = qvals.argmax().item() + 1

            # step environment
            next_state, raw_r, done = env.step(action)

            # potential-based shaping
            # curr_phi = -largest_power_range(env.product)
            # shaping  = gamma * curr_phi - prev_phi
            # reward   = raw_r + shaping
            # prev_phi = curr_phi

            current_largest_power_range = largest_power_range(env.product)

            if raw_r != 0.0:
                if raw_r > 0:
                    with open('identity.txt', 'a') as file:
                        file.write(env.render() + '\n')
                reward = raw_r
            else:
                if prev_largest_power_range > current_largest_power_range:
                    reward = 2
                elif prev_largest_power_range < current_largest_power_range:
                    reward = -1
                else:
                    reward = 0

            total_reward += reward

            prev_largest_power_range = current_largest_power_range

            # store transition
            replay_buffer.push(state, action-1, reward, next_state, done)
            state = next_state
            steps_done += 1

            # learn from batch
            if len(replay_buffer) >= batch_size:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(batch_size)
                s_b_long  = s_b.long()
                ns_b_long = ns_b.long()

                q_p = policy_net(s_b_long).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_n = target_net(ns_b_long).max(1)[0]
                    q_t = r_b + gamma * q_n * (1 - d_b)

                loss = F.mse_loss(q_p, q_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum   += loss.item()
                loss_count += 1

            # update target network
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

            # track average loss
            avg_loss_history.append((loss_sum / loss_count) if loss_count else 0.0)

        # after episode ends
        final_word  = env.render()
        final_range = largest_power_range(env.product)

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # record for logging
        recent_rewards.append(total_reward)
        recent_losses.append((loss_sum / loss_count) if loss_count else 0.0)

        # print every 100 episodes
        if episode % 100 == 0:
            avg_r      = sum(recent_rewards) / len(recent_rewards)
            avg_l      = sum(recent_losses)  / len(recent_losses)
            total      = win_count + lose_count + timeout_count
            win_pct    = win_count     / total * 100
            lose_pct   = lose_count    / total * 100
            timeout_pct= timeout_count / total * 100

            print(
                f"[Episode {episode:6d}] "
                f"AvgReward={avg_r:.3f}, "
                f"AvgLoss={avg_l:.5f}, "
                f"FinalWord={final_word}, "
                f"MaxPowerRange={final_range}, "
                f"Wins={win_pct:.0f}%, "
                f"Loses={lose_pct:.0f}%, "
                f"Timeouts={timeout_pct:.0f}%"
            )

            win_count = lose_count = timeout_count = 0

        # evaluate every 1000 episodes
        if episode % 1000 == 0:
            eval_env    = BurauEnv(max_steps, modulo)
            s_eval      = eval_env.reset()
            eval_ranges = []
            for _ in range(max_steps):
                eval_ranges.append(largest_power_range(eval_env.product))
                with torch.no_grad():
                    inp_eval  = torch.tensor(s_eval, dtype=torch.long, device=device).unsqueeze(0)
                    qvals_eval= policy_net(inp_eval).squeeze(0)
                a_eval, done_eval = qvals_eval.argmax().item() + 1, False
                s_eval, _, done_eval = eval_env.step(a_eval)
                if done_eval:
                    break
            eval_word = eval_env.render()
            plot_word_power_range(
                eval_word, eval_ranges,
                file_name=f'power_range_episode_{episode}.png'
            )

    # plot training loss
    plot_epochs_avg_loss(avg_loss_history)


class MCTSNode_MCTS:
    def __init__(self, env_state, parent=None, action=None):
        self.env_state = env_state  # a deep copy of BurauEnv
        self.parent = parent        # parent node
        self.action = action        # action taken from parent to this node
        self.children = {}          # action -> child node
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_fully_expanded(self):
        return set(self.children.keys()) == set(self.env_state.legal_actions())

    def uct_score(self, child, c_puct=1.0):
        # UCT: Q + c * sqrt(ln(N_parent) / N_child)
        if child.visits == 0:
            return float('inf')
        return child.value + c_puct * math.sqrt(math.log(self.visits) / child.visits)

    def select_child(self):
        # Select child with highest UCT score
        return max(self.children.values(), key=lambda n: self.uct_score(n))

    def expand(self):
        # Expand one untried action
        legal = self.env_state.legal_actions()
        for a in legal:
            if a not in self.children:
                env_copy = copy.deepcopy(self.env_state)
                state, raw_r, done = env_copy.step(a)
                child = MCTSNode_MCTS(env_copy, parent=self, action=a)
                self.children[a] = child
                return child
        return None

    def backup(self, value):
        # Backpropagate value
        node = self
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


def select_action_MCTS_MCTS(root_env, policy_net, device, num_sims=50, c_puct=1.0):
    # Build root node
    root = MCTSNode_MCTS(copy.deepcopy(root_env))
    for _ in range(num_sims):
        node = root
        # Selection
        while node.children and node.is_fully_expanded():
            node = node.select_child()
        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()
        # Evaluation: use Q-network to estimate state value
        state_vec = torch.tensor(node.env_state._get_state(), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qvals = policy_net(state_vec).cpu().numpy().squeeze()
            value = float(np.max(qvals))
        # Backup
        node.backup(value)
    # Choose action with highest visit count
    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action


def train_fc_MCTS(
        modulo,
        max_steps,
        num_episodes=100000,
        batch_size=64,
        gamma=0.99,
        lr=3e-4,
        buffer_capacity=100000,
        target_update_freq=1000,
        num_sims=50,
        c_puct=1.0,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999
    ):
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_net = BurauDQN(input_dim=max_steps).to(device)
    target_net = BurauDQN(input_dim=max_steps).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity, device)
    steps_done = 0

    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    avg_loss_history = []
    win_count = lose_count = timeout_count = 0

    epsilon = epsilon_start

    for episode in range(1, num_episodes + 1):
        env = BurauEnv(max_steps, modulo)
        state = env.reset()
        total_reward = 0.0
        loss_sum = loss_count = 0
        prev_max = largest_power_range(env.product)

        for t in range(max_steps):
            # MCTS-guided action selection with occasional random exploratory moves
            if random.random() < epsilon:
                action = random.choice(env.legal_actions())
            else:
                action = select_action_MCTS_MCTS(env, policy_net, device,
                                                 num_sims=num_sims, c_puct=c_puct)
            # Step
            next_state, raw_r, done = env.step(action)
            curr_max = largest_power_range(env.product)

            # Reward based on power-range change
            if raw_r != 0.0:
                reward = raw_r
            else:
                reward = 1.0 if curr_max < prev_max else -1.0 if curr_max > prev_max else 0.0
            prev_max = curr_max
            total_reward += reward

            # Store and train
            replay_buffer.push(state, action-1, reward, next_state, done)
            state = next_state
            steps_done += 1
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
                loss_sum += loss.item(); loss_count += 1

            # Update target network
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                if raw_r > 0: win_count += 1
                elif raw_r < 0: lose_count += 1
                else: timeout_count += 1
                break

            avg_loss_history.append(loss_sum/loss_count if loss_count else 0.0)

        # Logging
        recent_rewards.append(total_reward)
        recent_losses.append(loss_sum/loss_count if loss_count else 0.0)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 100 == 0:
            avg_r = sum(recent_rewards)/len(recent_rewards)
            avg_l = sum(recent_losses)/len(recent_losses)
            total = win_count+lose_count+timeout_count
            print(f"[Episode {episode:6d}] AvgR={avg_r:.2f}, AvgL={avg_l:.4f}, "
                  f"Win%={win_count/total*100:.0f}, Loss%={lose_count/total*100:.0f}, Timeout%={timeout_count/total*100:.0f}")
            win_count = lose_count = timeout_count = 0

    plot_epochs_avg_loss(avg_loss_history)
