from Config import Config
import numpy as np
import torch

from Utils import to_tensor


class Noiser:
    def __init__(self, action_space, config: Config):
        self.theta = config.theta
        self.sigma = config.sigma
        self.action_dim = action_space.shape[0]
        self.low = action_space.low[0]
        self.high = action_space.high[0]
        self.eps_max = config.eps_max
        self.eps_min = config.eps_min
        self.eps = self.eps_max
        self.eps_decay = (self.eps_max - self.eps_min) / config.max_episodes
        self.state = np.zeros(self.action_dim)
        self.device = config.device

    def update_eps(self):
        self.eps -= self.eps_decay

    def apply(self, action, clip=False):
        self.state += - self.theta * self.state + self.sigma * np.random.randn(self.action_dim)
        noise = self.eps * self.state
        noise = np.clip(noise, -2 * self.sigma, 2 * self.sigma) if clip else noise
        if isinstance(action, torch.Tensor):
            return torch.clamp(action + to_tensor(noise, device=self.device), self.low, self.high)
        else:
            return np.clip(action + noise, self.low, self.high)
