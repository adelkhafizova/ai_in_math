import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class QNetwork(nn.Module):
    """
    Dueling LSTM Q-network.
    - Shared LSTM feature extractor over token sequence.
    - Two heads:
        * Value head V(s) -> (B, 1)
        * Advantage head A(s, a) -> (B, output_dim)
    - Combine: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    Constructor is strict and mirrors your LSTM signature so Data.model_params stays the same.
    """
    def __init__(
        self,
        *,
        input_size: int,      # features per time-step (you use 1)
        hidden_dim: int,
        num_layers: int,
        output_dim: int,      # number of actions (4)
        bidirectional: bool,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.value_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        for m in [self.value_head[-1], self.adv_head[-1]]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, lengths):
        # Accept (T,), (B, T), or (B, T, 1); convert to (B, T, 1)
        if x.dim() == 1:
            x = x.unsqueeze(0)           # (T,) -> (1, T)
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # (B, T) -> (B, T, 1)

        # lengths must be CPU LongTensor (your Trainer already ensures this)
        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]                 # (B, lstm_out_dim)

        v = self.value_head(h_last)      # (B, 1)
        a = self.adv_head(h_last)        # (B, output_dim)

        a_mean = a.mean(dim=1, keepdim=True)  # (B, 1)
        q = v + (a - a_mean)             # (B, output_dim)

        return q