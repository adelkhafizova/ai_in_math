import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
            self,
            *,
            input_dim,
            hidden1,
            hidden2,
            output_dim
    ):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)