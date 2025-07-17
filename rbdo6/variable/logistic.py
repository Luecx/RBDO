# ============================================================
# File        : logistic.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Logistic random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Logistic(RandomVariable):
    """
    Logistic random variable with location and scale parameters.

    Standard normal samples z_i are transformed to uniform values via the normal CDF,
    and then mapped to logistic samples via the inverse CDF:
        X = loc + scale * log(U / (1 - U))

    Attributes:
        loc (DesignVariable or float): Location parameter (mean).
        scale (DesignVariable or float): Scale parameter (> 0).
    """

    def __init__(self, loc, scale):
        """
        Initializes a Logistic random variable and registers it with the context.

        Args:
            loc (float or DesignVariable): Location parameter.
            scale (float or DesignVariable): Scale parameter.
        """
        super().__init__()
        self.loc = loc
        self.scale = scale

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a logistic distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from Logistic(loc, scale).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        loc = self.get_value(self.loc, v)
        scale = self.get_value(self.scale, v)
        return loc + scale * torch.log(U / (1 - U))

