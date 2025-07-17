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

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a uniform distribution using inverse CDF.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from U(a, b).
        """
        a = self.get_value(self.a, v)
        b = self.get_value(self.b, v)
        u_std = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0, dtype=z_i.dtype, device=z_i.device))))
        return a + u_std * (b - a)
