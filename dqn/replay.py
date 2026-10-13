import random
from collections import deque

import numpy as np
import torch

class ReplayBuffer:

    def __init__(self, capacity: int):

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        def to_t(arr, dtype):
            return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)

        return (
            to_t(states, torch.float32),
            to_t(actions, torch.int64).unsqueeze(1),
            to_t(rewards, torch.float32),
            to_t(next_states, torch.float32),
            to_t(dones, torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
