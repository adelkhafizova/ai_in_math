import random
import torch


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