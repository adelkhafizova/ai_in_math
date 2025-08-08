import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Dueling_LSTM(nn.Module):
    """
    Dueling LSTM-based Q-network:
    - Packs variable-length sequences
    - Bidirectional optional
    - Splits into Value and Advantage streams
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_actions: int = 4,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Value stream: outputs single state-value
        self.value_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream: outputs advantage for each action
        self.advantage_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len) or (batch, seq_len, input_dim)
            lengths: 1D LongTensor of sequence lengths (batch,)
        Returns:
            Q-values: Tensor of shape (batch, num_actions)
        """
        # Ensure input has feature dim
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (T,) -> (1, T)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        # Pack padded sequence; lengths must be on CPU
        lengths = lengths.view(-1)
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)

        # Take last layer's hidden state
        # Index = last layer * num_directions to last layer * num_directions + directions
        last_layer_index = (self.num_layers - 1) * self.num_directions
        h_last = h_n[last_layer_index:last_layer_index + self.num_directions]
        # Concatenate directions if bidirectional
        h_last = h_last.transpose(0, 1).contiguous().view(x.size(0), -1)
        # h_last shape: (batch, hidden_dim * num_directions)

        # Compute Value and Advantage
        V = self.value_stream(h_last)         # (batch, 1)
        A = self.advantage_stream(h_last)     # (batch, num_actions)

        # Combine into Q-values: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        A_mean = A.mean(dim=1, keepdim=True)
        Q = V + (A - A_mean)
        return Q