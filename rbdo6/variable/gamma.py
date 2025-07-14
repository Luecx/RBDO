# ============================================================
# File        : gamma.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Gamma random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Gamma(RandomVariable):
    """
    Gamma random variable with concentration and rate parameters.

    Transforms standard normal samples into Gamma-distributed values
    via inverse CDF (percent-point function) sampling.

    Attributes:
        concentration (DesignVariable or float): Shape parameter (α > 0).
        rate (DesignVariable or float): Rate parameter (β > 0).
    """

    def __init__(self, concentration, rate):
        """
        Initializes a Gamma random variable and registers it with the context.

        Args:
            concentration (float or DesignVariable): Shape parameter α.
            rate (float or DesignVariable): Rate parameter β.
        """
        super().__init__()
        self.concentration = concentration
        self.rate = rate

    def sample(self, z_i):
        """
        Samples from the Gamma distribution using inverse transform sampling.

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples from the Gamma(α, β) distribution (shape [B]).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        concentration = self.get_value(self.concentration)
        rate = self.get_value(self.rate)
        dist = torch.distributions.Gamma(concentration, rate)
        return dist.icdf(U)
