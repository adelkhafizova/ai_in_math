import torch.nn as nn
import torch.nn.functional as F


class BurauDQN(nn.Module):
    def __init__(self, input_size):
        super(BurauDQN, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        # if isinstance(x, np.ndarray):
            # x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # make batch of 1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)  # raw Q-values for each action