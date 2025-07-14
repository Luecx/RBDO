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

    def sample(self, z_i):
        """
        Samples from the Logistic distribution using inverse transform sampling.

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples from the Logistic distribution (shape [B]).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        loc = self.get_value(self.loc)
        scale = self.get_value(self.scale)
        return loc + scale * torch.log(U / (1 - U))
