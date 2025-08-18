import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class QNetwork(nn.Module):
    """
    New LSTM-based network for longer words / higher moduli.
    - Accepts input either as (batch, seq_len) or (batch, seq_len, features).
      • If 2-D, it is automatically unsqueezed to (batch, seq_len, 1).
    - Outputs Q-values of shape (batch, output_dim) compatible with DQN loops.
    """
    def __init__(
        self,
        *,
        input_size: int, # features per time-step
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        bidirectional: bool,
    ):
        super(QNetwork, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_out_dim, output_dim)
        )

    def forward(self, x, lengths):
        # handle 1-D and 2-D inputs:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (T,) -> (1, T)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]            # (B, hidden)
        q_out  = self.head(h_last)

        return q_out