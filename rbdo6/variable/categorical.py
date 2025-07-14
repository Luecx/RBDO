# ============================================================
# File        : categorical.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Categorical random variable using
#               inverse transform sampling from standard normal input.
# ============================================================

import torch
from .random import RandomVariable
from .design import DesignVariable


class Categorical(RandomVariable):
    """
    Categorical random variable representing a discrete probability distribution.

    Transforms standard normal samples to category indices using inverse transform sampling.
    The input z_i is mapped to uniform values and compared against the cumulative probability
    mass function to determine sampled indices.

    Attributes:
        probs (DesignVariable or torch.Tensor): A [B, N] tensor of category probabilities per sample.
    """

    def __init__(self, probs):
        """
        Initializes a Categorical random variable and registers it with the context.

        Args:
            probs (DesignVariable or torch.Tensor): Tensor of category probabilities (shape [B, N]).
        """
        super().__init__()
        self.probs = probs

    def sample(self, z_i):
        """
        Samples category indices from the categorical distribution using inverse transform sampling.

        Args:
            z_i (torch.Tensor): Standard normal samples (shape [B]).

        Returns:
            torch.Tensor: Sampled category indices (shape [B]).
        """
        U = 0.5 * (1 + torch.erf(z_i / torch.sqrt(torch.tensor(2.0))))  # Φ(z_i)
        probs = self.get_value(self.probs)  # [B, N]
        cdf = torch.cumsum(probs, dim=1)    # [B, N]
        sample = (U.unsqueeze(-1) < cdf).float()  # [B, N]
        return torch.argmax(sample, dim=1)
