# ============================================================
# File        : chisquare.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Chi-Square random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class ChiSquare(RandomVariable):
    """
    Chi-Square random variable with degrees of freedom ν.

    Transforms standard normal samples into Chi-Square distributed values
    via inverse CDF (quantile function) sampling.

    Attributes:
        df (DesignVariable or float): Degrees of freedom (ν > 0).
    """

    def __init__(self, df):
        """
        Initializes a Chi-Square random variable and registers it with the context.

        Args:
            df (float or DesignVariable): Degrees of freedom.
        """
        super().__init__()
        self.df = df

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a chi-squared distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from Chi2(df).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        df = self.get_value(self.df, v)
        dist = torch.distributions.Chi2(df)
        return dist.icdf(U)
