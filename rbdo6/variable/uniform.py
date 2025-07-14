# ============================================================
# File        : uniform.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Uniform random variable using
#               inverse transform sampling from standard normal samples.
# ============================================================

import torch
from torch.special import erf
from .random import RandomVariable
from .design import DesignVariable


class Uniform(RandomVariable):
    """
    Uniform random variable U(a, b), supporting both fixed bounds
    and design-dependent parameters.

    Standard normal samples z_i ∈ N(0,1) are transformed to uniform
    samples in [a, b] using inverse transform sampling:
        1. u = Φ(z_i) maps z_i to [0, 1] using the standard normal CDF
        2. x = a + u * (b - a) maps to [a, b]

    Attributes:
        a (DesignVariable or float): Lower bound of the distribution.
        b (DesignVariable or float): Upper bound of the distribution.
    """

    def __init__(self, a, b):
        """
        Initializes a Uniform random variable and registers it with the context.

        Args:
            a (float or DesignVariable): Lower bound of the distribution.
            b (float or DesignVariable): Upper bound of the distribution.
        """
        super().__init__()
        self.a = a
        self.b = b

    def sample(self, z_i: torch.Tensor) -> torch.Tensor:
        """
        Transforms standard normal samples into uniform [a, b] samples.

        This uses the inverse transform sampling approach:
            u = Φ(z_i) = 0.5 * (1 + erf(z_i / sqrt(2)))
            x = a + u * (b - a)

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples from Uniform(a, b) (shape [B]).
        """
        a = self.get_value(self.a)
        b = self.get_value(self.b)

        # Compute standard normal CDF
        u = 0.5 * (1.0 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0, dtype=z_i.dtype, device=z_i.device))))
        return a + u * (b - a)
