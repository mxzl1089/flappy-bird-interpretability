import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):

    def __init__(self, num_actions: int = 2, in_channels: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m is self.actor:
                gain = 0.01
            elif m is self.critic:
                gain = 1.0
            else:
                gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        h = self.shared(self.features(x))
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy(), value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
