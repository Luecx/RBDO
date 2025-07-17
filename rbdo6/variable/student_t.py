# ============================================================
# File        : studentt.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Student-t random variable using
#               inverse CDF sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class StudentT(RandomVariable):
    """
    Student-t random variable with support for design-dependent parameters.

    Transforms standard normal samples `z_i` into Student-t distributed values
    via the inverse CDF (percent point function) method.

    Attributes:
        df (DesignVariable or float): Degrees of freedom (ν > 0).
        loc (DesignVariable or float): Location parameter (mean).
        scale (DesignVariable or float): Scale parameter (> 0).
    """

    def __init__(self, df, loc, scale):
        """
        Initializes a Student-t random variable and registers it with the context.

        Args:
            df (float or DesignVariable): Degrees of freedom.
            loc (float or DesignVariable): Location (mean).
            scale (float or DesignVariable): Scale parameter.
        """
        super().__init__()
        self.df = df
        self.loc = loc
        self.scale = scale

    def sample(self, z_i, v=None):
        """
        Transforms standard normal input to a Student-t distribution.

        Args:
            z_i (Tensor): Standard normal input.
            v (Tensor): Design variable values.

        Returns:
            Tensor: Sample from StudentT(df, loc, scale).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))
        df = self.get_value(self.df, v)
        loc = self.get_value(self.loc, v)
        scale = self.get_value(self.scale, v)
        dist = torch.distributions.StudentT(df, loc, scale)
        return dist.icdf(U)

