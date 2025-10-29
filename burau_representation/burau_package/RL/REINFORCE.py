import gymnasium as gym
import numpy as np
import typing as tt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

State = int
Action = int
Reward = int

class PG_Agent(nn.Module):
    def __init__(self, env, input_size: int, n_actions: int):
        super(PG_Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.env = env
        self.state = self.env.reset()[0]

    def make_rand_step(self) -> tt.Tuple[State, Action, Reward, State]:
        '''
        Chooose a random action, make one step and return the result
        '''
        state = self.state
        action = self.select_action()
        new_state, reward, is_done, is_trunc, _ = \
                self.env.step(action)
        if is_done or is_trunc:
            self.state, _ = self.env.reset()
            new_state = None
        else:
            self.state = new_state
        return state, action, reward, new_state

    def select_action(self) -> Action:
        '''
        Chooose action with prob given by the policy
        '''
        state = torch.Tensor(self.state)
        logits_v = self.net(state)
        probs_v = F.softmax(logits_v, dim=-1)
        probs = probs_v.data.cpu().numpy()
        action = np.random.choice(len(probs), p=probs)
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def calc_qvals(rewards: tt.List[float], baseline=False) -> tt.List[float]:
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    if baseline:
        mean_q = np.mean(res)
        return [q - mean_q for q in res]
    return res


def reinforce(env, baseline=False, GAMMA=GAMMA,
              LEARNING_RATE = LEARNING_RATE, EPISODES_TO_TRAIN=EPISODES_TO_TRAIN,
              ):

    agent = PG_Agent(env, env.observation_space.shape[0], env.action_space.n)
    print(agent.net)

    optimizer = optim.Adam(agent.net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []


    step_idx = 0
    while True:
        # print("step", step_idx)
        total_reward = None
        step_idx += 1

        '''
        Step 2. Play full episodes, saving their (s,a,r,s') transitions.
        '''
        state, action, reward, new_state = agent.make_rand_step()
        batch_states.append(state)
        batch_actions.append(action)
        cur_rewards.append(reward)

        if new_state is None:
            '''
            Step 3. Calculate the discounted total reward
            '''
            batch_qvals.extend(calc_qvals(cur_rewards))
            total_reward = sum(cur_rewards)
            cur_rewards.clear()
            batch_episodes += 1


        '''
        Save total rewards at the end of episodes
        Print data every 50 episodes
        '''
        if total_reward:
            done_episodes += 1
            total_rewards.append(total_reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            if (done_episodes % 50 == 0):
                print(f"{step_idx}: reward: {total_reward:6.2f}, mean_100: {mean_rewards:6.2f}, "
                  f"episodes: {done_episodes}")
            if mean_rewards > 450:
                print(f"Solved in {step_idx} steps and {done_episodes} episodes!")
                break

        '''
        Making a batch of episodes to train the agent
        '''
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        states_t = torch.as_tensor(np.asarray(batch_states))
        batch_actions_t = torch.as_tensor(np.asarray(batch_actions))
        batch_qvals_t = torch.as_tensor(np.asarray(batch_qvals))


        '''
        Step 4. Calculate loss L = -\sum (Q \log \pi) and backpropagate
        '''
        logits_t = agent(states_t.float()) # Predictions of NN
        log_prob_t = F.log_softmax(logits_t, dim=1) # log(\pi(s_{k,t}, a_{k,t}))
        batch_idx = range(len(batch_states))
        act_probs_t = log_prob_t[batch_idx, batch_actions_t]
        log_prob_actions_v = batch_qvals_t * act_probs_t
        loss_t = -log_prob_actions_v.mean()

        loss_t.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()