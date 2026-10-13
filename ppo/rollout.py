import numpy as np
import torch
from collections import namedtuple

Minibatch = namedtuple("Minibatch", [
    "states",
    "actions",
    "old_logprobs",
    "old_values",
    "returns",
    "advantages",
])

class RolloutBuffer:

    def __init__(self, n_steps: int, state_shape, num_actions: int = 2):
        self.n_steps = n_steps
        self.state_shape = state_shape

        self.states = np.zeros((n_steps, *state_shape), dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.logprobs = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)

        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        self.pos = 0

    def reset(self):
        self.pos = 0

    def add(self, state, action, reward, done, logprob, value):
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.logprobs[i] = logprob
        self.values[i] = value
        self.pos += 1

    @property
    def full(self) -> bool:
        return self.pos >= self.n_steps

    def compute_returns_and_advantages(self, last_value: float,
                                       gamma: float, lam: float,
                                       normalize: bool = True):
        gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:

                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:

                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        if normalize:

            self.advantages = (
                (self.advantages - self.advantages.mean())
                / (self.advantages.std() + 1e-8)
            )

    def get_minibatches(self, batch_size: int, n_epochs: int, device):

        t_states = torch.as_tensor(self.states, device=device)
        t_actions = torch.as_tensor(self.actions, device=device)
        t_logprobs = torch.as_tensor(self.logprobs, device=device)
        t_values = torch.as_tensor(self.values, device=device)
        t_returns = torch.as_tensor(self.returns, device=device)
        t_advs = torch.as_tensor(self.advantages, device=device)

        indices = np.arange(self.n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.n_steps, batch_size):
                idx = indices[start:start + batch_size]
                idx_t = torch.as_tensor(idx, device=device)
                yield Minibatch(states=t_states[idx_t],
                                actions=t_actions[idx_t],
                                old_logprobs=t_logprobs[idx_t],
                                old_values=t_values[idx_t],
                                returns=t_returns[idx_t],
                                advantages=t_advs[idx_t])
