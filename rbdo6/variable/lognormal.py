# ============================================================
# File        : lognormal.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Lognormal random variable using
#               exponential transformation of standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class LogNormal(RandomVariable):
    """
    Lognormal random variable defined by parameters μ and σ of the underlying normal distribution.

    The transformation is applied as:
        X = exp(μ + σ * z),  with z ~ N(0, 1)

    Attributes:
        mu (DesignVariable or float): Mean of the underlying normal distribution.
        sigma (DesignVariable or float): Standard deviation of the underlying normal distribution.
    """

    def __init__(self, mu, sigma):
        """
        Initializes a Lognormal random variable and registers it with the context.

        Args:
            mu (float or DesignVariable): Mean of the underlying normal distribution.
            sigma (float or DesignVariable): Standard deviation.
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a lognormal distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values [B, n_v].

        Returns:
            Tensor: Sample from LogNormal(mu, sigma).
        """
        mu = self.get_value(self.mu, v)
        sigma = self.get_value(self.sigma, v)
        return torch.exp(mu + sigma * z_i)

