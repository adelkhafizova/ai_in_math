import random
import torch


class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, max_steps, alpha=0.6, eps=1e-6):
        """
        capacity: max number of transitions
        device: torch device for tensors
        max_steps: unused here, but kept for compatibility
        alpha: exponent for priorities (0 = uniform, 1 = full prioritization)
        eps: small constant to ensure non-zero priority
        """
        self.capacity = capacity
        self.device = device
        self.max_steps = max_steps
        self.alpha = alpha
        self.eps = eps

        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, state_length,
             next_state, next_state_length, done):
        # wrap scalars as 1‑element tensors on the right device:
        a = torch.tensor([action], dtype=torch.int64, device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        d = torch.tensor([done], dtype=torch.float32, device=self.device)
        # state_length and next_state_length are already tensors

        transition = (state, a, r,
                      state_length, next_state, next_state_length, d)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(1.0)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max(self.priorities)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Returns:
          batch: tuple of (states, actions, rewards, state_lengths,
                           next_states, next_state_lengths, dones)
          idxs: list of sampled indices
          weights: Tensor of shape (batch_size,) on self.device
        """
        if len(self.buffer) == 0:
            raise ValueError("Trying to sample from empty buffer")

        # compute probabilities
        prios = torch.tensor(self.priorities, dtype=torch.float32)
        probs = prios.pow(self.alpha)
        P = probs / probs.sum()

        # sample indices with replacement
        idxs = random.choices(
            population=range(len(self.buffer)),
            weights=P.tolist(),
            k=batch_size
        )

        # fetch the transitions
        batch = [ self.buffer[i] for i in idxs ]
        states, actions, rewards, state_lengths, next_states, next_state_lengths, dones = zip(*batch)

        # stack into tensors
        batch_states      = torch.stack(states)
        batch_next_states = torch.stack(next_states)
        actions           = torch.cat(actions)
        rewards           = torch.cat(rewards)
        state_lengths     = torch.cat(state_lengths)
        next_state_lengths= torch.cat(next_state_lengths)
        dones             = torch.cat(dones)

        # importance-sampling weights
        N = len(self.buffer)
        P_i = P[idxs]
        weights = (N * P_i).pow(-beta)
        weights = weights / weights.max()
        weights = weights.to(self.device)

        return (
            (batch_states, actions, rewards,
             state_lengths, batch_next_states,
             next_state_lengths, dones),
            idxs,
            weights
        )

    def update_priorities(self, idxs, td_errors):
        """
        idxs: list of indices returned by sample()
        td_errors: numpy array or list of absolute TD errors
        """
        for idx, err in zip(idxs, td_errors):
            # ensure non-zero priority
            self.priorities[idx] = abs(err) + self.eps

    def __len__(self):
        return len(self.buffer)