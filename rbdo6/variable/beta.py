# ============================================================
# File        : beta.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Beta random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Beta(RandomVariable):
    """
    Beta random variable with shape parameters α and β.

    Transforms standard normal samples into Beta-distributed values using
    inverse transform sampling via the Beta distribution's CDF.

    Attributes:
        alpha (DesignVariable or float): First shape parameter (α > 0).
        beta (DesignVariable or float): Second shape parameter (β > 0).
    """

    def __init__(self, alpha, beta):
        """
        Initializes a Beta random variable and registers it with the context.

        Args:
            alpha (float or DesignVariable): Shape parameter α.
            beta (float or DesignVariable): Shape parameter β.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def sample(self, z_i):
        """
        Samples from the Beta distribution using inverse transform sampling.

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples from the Beta(α, β) distribution (shape [B]).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))  # Φ(z_i)
        alpha = self.get_value(self.alpha)
        beta  = self.get_value(self.beta)
        dist = torch.distributions.Beta(alpha, beta)
        return dist.icdf(U)
