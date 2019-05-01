import numpy as np
import copy
from collections import namedtuple, deque

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self, decay=None, sigma_min=None):
        """Reset the internal state (= noise) to mean (mu)."""
        if decay is not None and sigma_min is not None:
            self.sigma = max((1-decay) * self.sigma, sigma_min)
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state