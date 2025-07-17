# ============================================================
# File        : normal.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Normal (Gaussian) random variable
#               using a linear transformation of standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Normal(RandomVariable):
    """
    Normal (Gaussian) random variable N(μ, σ), supporting both
    constant and design-dependent parameters.

    Attributes:
        mu (DesignVariable or float): Mean of the distribution.
        sigma (DesignVariable or float): Standard deviation (σ > 0).
    """

    def __init__(self, mu, sigma):
        """
        Initializes a Normal random variable and registers it with the context.

        Args:
            mu (float or DesignVariable): Mean value.
            sigma (float or DesignVariable): Standard deviation.
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a normal distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values [B, n_v].

        Returns:
            Tensor: Sample from N(mu, sigma).
        """
        mu = self.get_value(self.mu, v)
        sigma = self.get_value(self.sigma, v)
        return mu + sigma * z_i
