# ============================================================
# File        : poisson.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Poisson random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Poisson(RandomVariable):
    """
    Poisson random variable with rate λ.

    Transforms standard normal samples into Poisson-distributed integer values
    using the inverse CDF (quantile) method via the torch.distributions API.

    Attributes:
        rate (DesignVariable or float): Rate parameter λ (> 0).
    """

    def __init__(self, rate):
        """
        Initializes a Poisson random variable and registers it with the context.

        Args:
            rate (float or DesignVariable): Expected number of events (λ).
        """
        super().__init__()
        self.rate = rate

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a Poisson distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from Poisson(rate).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        rate = self.get_value(self.rate, v)
        dist = torch.distributions.Poisson(rate)
        return dist.icdf(U)

