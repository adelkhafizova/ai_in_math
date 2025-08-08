import random

import torch


class ReplayBuffer:
    def __init__(self, capacity, device, max_steps):
        self.capacity = capacity
        self.device = device
        self.max_steps = max_steps
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, state_length, next_state, next_state_length, done):
        # Convert to CPU tensors
        s = state
        a = torch.tensor([action], dtype=torch.int64, device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        sl = state_length
        ns = next_state
        nsl = next_state_length
        d = torch.tensor([done], dtype=torch.float32, device=self.device)

        # Only append placeholder when buffer isn't full yet
        if len(self.buffer) < self.capacity:
            self.buffer.append((None, None, None, None, None, None, None))

        # Store transition at current position
        self.buffer[self.position] = (s, a, r, sl, ns, nsl, d)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions; returned tensors are ready on the correct device.
        States are already fixed-length from push().
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, state_lengths, next_states, next_state_lengths, dones = zip(*batch)

        # Stack fixed-size tensors
        batch_states = torch.stack(states)
        batch_next_states = torch.stack(next_states)

        # Stack scalar tensors
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        state_lengths = torch.cat(state_lengths)
        next_state_lengths = torch.cat(next_state_lengths)
        dones   = torch.cat(dones)

        return batch_states, actions, rewards, state_lengths, batch_next_states, next_state_lengths, dones

    def __len__(self):
        return len(self.buffer)