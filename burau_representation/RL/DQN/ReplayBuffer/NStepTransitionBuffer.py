from collections import deque


class NStepTransitionBuffer:
    def __init__(self, n_step, gamma):
        """
        Collect up to n_step transitions and then emit
        an aggregated n‑step transition via get().
        """
        self.n_step = n_step
        self.gamma  = gamma
        self.buffer = deque()  # will hold tuples of (s, a, r, sl, sn, sln, done)

    def push(self, state, action, reward, state_length,
             next_state, next_state_length, done):
        """
        Add a single 1‑step transition to the buffer.
        We keep at most n_step entries so that the window
        always represents the last n_step transitions.
        """
        self.buffer.append((state, action, reward,
                            state_length, next_state,
                            next_state_length, done))
        # If we exceed n steps, pop the oldest so the window slides
        if len(self.buffer) > self.n_step:
            self.buffer.popleft()

    def get(self):
        """
        Once buffer has n_step entries, compute:
           R = sum_{i=0..n-1} gamma^i * r_{t+i}
        and return the aggregated transition:
           (s_t, a_t, R, sl_t, s_{t+n}, sl_{t+n}, done_{t+n})
        If there aren’t yet n_step entries, returns None.
        """
        if len(self.buffer) < self.n_step:
            return None

        # discounted return over the window
        R = 0.0
        for idx, (_, _, r, _, _, _, _) in enumerate(self.buffer):
            R += (self.gamma ** idx) * r

        # first and last elements
        s0, a0, _, sl0, _, _, _ = self.buffer[0]
        _,   _, _, _, sn, sln, dn = self.buffer[-1]

        return (s0, a0, R, sl0, sn, sln, dn)

    def __len__(self):
        return len(self.buffer)