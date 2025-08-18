import random
import torch


class ReplayBuffer:
    """
    CPU-based replay buffer with compact storage:
      - seq tensors stored as torch.uint8 on CPU (values 0..4), optionally pinned
      - lengths kept on CPU (LongTensor) for pack_padded_sequence
      - masks stored as CPU bool
      - scalar fields (action, reward, done) stored on CPU
    On sample(), batches are moved to `device` with non_blocking=True and
    seq tensors are cast to float32 ON DEVICE to avoid extra host work.
    """
    def __init__(self, capacity, device, max_steps, pin_memory: bool = True):
        self.capacity = capacity
        self.device = device
        self.max_steps = max_steps
        self.pin_memory = pin_memory

        self.buffer = []
        self.position = 0

    def _maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        if self.pin_memory and t.device.type == "cpu":
            try:
                return t.pin_memory()
            except RuntimeError:
                # Pinned memory may be unavailable in some envs; fall back silently
                return t
        return t

    def push(
        self,
        state,                # (T,) float on device; values 0..4
        action,               # int (0..3)
        reward,               # float
        state_length,         # (1,) long on CPU
        next_state,           # (T,) float on device
        next_state_length,    # (1,) long on CPU
        terminal_for_target,  # float 0/1
        next_action_mask      # (4,) bool on device or CPU
    ):
        # Move sequences to CPU and store as uint8 (0..4)
        s = state.detach().to("cpu", copy=True).to(torch.uint8)
        ns = next_state.detach().to("cpu", copy=True).to(torch.uint8)

        # Lengths stay on CPU (needed by pack_padded_sequence)
        sl = state_length.detach().to("cpu", copy=True).to(torch.long)
        nsl = next_state_length.detach().to("cpu", copy=True).to(torch.long)

        # Scalars on CPU
        a = torch.tensor([action], dtype=torch.int64)          # 0..3
        r = torch.tensor([reward], dtype=torch.float32)
        t = torch.tensor([terminal_for_target], dtype=torch.float32)

        # Masks on CPU bool
        nm = torch.as_tensor(next_action_mask, dtype=torch.bool).detach().to("cpu", copy=True)

        # Optionally pin CPU tensors for faster H→D copies
        s = self._maybe_pin(s)
        ns = self._maybe_pin(ns)
        a = self._maybe_pin(a)
        r = self._maybe_pin(r)
        t = self._maybe_pin(t)
        nm = self._maybe_pin(nm)
        # sl/nsl remain on CPU; pinning not required since they are not transferred

        if len(self.buffer) < self.capacity:
            self.buffer.append((None,)*8)

        self.buffer[self.position] = (s, a, r, sl, ns, nsl, t, nm)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns:
          batch_states         (B, T) float32 on device
          actions              (B,)   int64   on device
          rewards              (B,)   float32 on device
          state_lengths        (B,)   long    on CPU
          batch_next_states    (B, T) float32 on device
          next_state_lengths   (B,)   long    on CPU
          terms                (B,)   float32 on device
          next_masks           (B, 4) bool    on device
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sl, ns, nsl, t, nm = zip(*batch)

        # Stack on CPU
        s = torch.stack(s)          # uint8 CPU
        ns = torch.stack(ns)        # uint8 CPU
        a = torch.cat(a)            # CPU
        r = torch.cat(r)            # CPU
        sl = torch.cat(sl)          # CPU (kept on CPU)
        nsl = torch.cat(nsl)        # CPU (kept on CPU)
        t = torch.cat(t)            # CPU
        nm = torch.stack(nm)        # CPU

        # Transfer to device; cast seq tensors to float32 ON DEVICE
        s = s.to(self.device, non_blocking=True).to(torch.float32)
        ns = ns.to(self.device, non_blocking=True).to(torch.float32)
        a = a.to(self.device, non_blocking=True)
        r = r.to(self.device, non_blocking=True)
        t = t.to(self.device, non_blocking=True)
        nm = nm.to(self.device, non_blocking=True)

        return s, a, r, sl, ns, nsl, t, nm

    def __len__(self):
        return len(self.buffer)