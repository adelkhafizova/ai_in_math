import torch.nn as nn


class DQN_LSTM(nn.Module):
    """
    New LSTM-based network for longer words / higher moduli.
    - Accepts input either as (batch, seq_len) or (batch, seq_len, features).
      • If 2-D, it is automatically unsqueezed to (batch, seq_len, 1).
    - Outputs Q-values of shape (batch, output_dim) compatible with DQN loops.
    """
    def __init__(
        self,
        input_size: int = 1,         # features per time-step
        hidden_dim: int = 128,
        num_layers: int = 1,
        output_dim: int = 4,
        bidirectional: bool = False,
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

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_out_dim, output_dim)
        )

    def forward(self, x):
        # Accept (batch, seq_len) or (batch, seq_len, features)
        if x.dim() == 2:             # (batch, seq_len) → add feature dim
            x = x.unsqueeze(-1)      # (batch, seq_len, 1)

        # LSTM returns (output, (h_n, c_n))
        out, _ = self.lstm(x)        # out: (batch, seq_len, hidden)
        last_step = out[:, -1, :]    # take final time-step representation

        return self.head(last_step)