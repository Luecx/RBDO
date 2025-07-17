# ============================================================
# File        : weibull.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Weibull random variable with
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Weibull(RandomVariable):
    """
    Weibull random variable defined by a scale and concentration parameter.

    The transformation from standard normal samples to physical values is
    performed via the inverse cumulative distribution function (inverse CDF).

    The standard normal samples `z_i` are first mapped to uniform samples via
    the standard normal CDF, then mapped to the Weibull distribution.

    Attributes:
        scale (DesignVariable or float): Scale parameter (λ > 0).
        concentration (DesignVariable or float): Shape parameter (k > 0).
    """

    def __init__(self, scale, concentration):
        """
        Initializes a Weibull random variable and registers it with the context.

        Args:
            scale (DesignVariable or float): Scale parameter (λ).
            concentration (DesignVariable or float): Shape parameter (k).
        """
        super().__init__()
        self.scale = scale
        self.concentration = concentration

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a Weibull distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from Weibull(scale, concentration).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        scale = self.get_value(self.scale, v)
        concentration = self.get_value(self.concentration, v)
        return scale * torch.pow(-torch.log(1 - U), 1.0 / concentration)

