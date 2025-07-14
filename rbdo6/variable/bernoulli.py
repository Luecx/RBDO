# ============================================================
# File        : bernoulli.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Bernoulli random variable using
#               inverse transform sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Bernoulli(RandomVariable):
    """
    Bernoulli random variable with success probability p.

    Standard normal samples are mapped to uniform values, then compared
    to p to generate binary outcomes in {0, 1}.

    Attributes:
        p (DesignVariable or float): Probability of success (0 ≤ p ≤ 1).
    """

    def __init__(self, p):
        """
        Initializes a Bernoulli random variable and registers it with the context.

        Args:
            p (float or DesignVariable): Probability of success.
        """
        super().__init__()
        self.p = p

    def sample(self, z_i):
        """
        Samples from the Bernoulli distribution using inverse transform sampling.

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Samples of 0 or 1 from the Bernoulli(p) distribution (shape [B]).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))  # Φ(z_i)
        p = self.get_value(self.p)
        return (U < p).float()
