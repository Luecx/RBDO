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

    def sample(self, z_i):
        """
        Samples from the Normal distribution using standard transformation:
        X = μ + σ * z_i, where z_i ~ N(0,1)

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples from N(μ, σ) (shape [B]).
        """
        mu = self.get_value(self.mu)
        sigma = self.get_value(self.sigma)
        return mu + sigma * z_i
