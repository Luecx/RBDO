# ============================================================
# File        : exponential.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Exponential random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Exponential(RandomVariable):
    """
    Exponential random variable with a rate parameter λ.

    Transforms standard normal samples into exponential values using
    inverse transform sampling:
        X = -ln(1 - U) / λ, where U = Φ(z)

    Attributes:
        rate (DesignVariable or float): Rate parameter (λ > 0).
    """

    def __init__(self, rate):
        """
        Initializes an Exponential random variable and registers it with the context.

        Args:
            rate (float or DesignVariable): Rate parameter λ.
        """
        super().__init__()
        self.rate = rate

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to an exponential distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from Exp(rate).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        rate = self.get_value(self.rate, v)
        return -torch.log(1 - U) / rate

