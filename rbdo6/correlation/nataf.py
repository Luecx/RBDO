# ============================================================
# File        : nataf.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Applies Nataf transformation to standard normal inputs.
#               Converts uncorrelated u → correlated z using a Cholesky
#               factor L such that z = Lᵀ u.
# ============================================================

import torch
from ..core import Node, Context


class NatafTransformation(Node):
    """
    Applies the Nataf transformation to convert uncorrelated u to correlated z.

    The transformation uses the Cholesky factor L of the correlation matrix:
        z = u @ L.T
    """

    def __init__(self, correlation_matrix):
        """
        Initializes the Nataf transformation and registers UNode as input.

        Args:
            correlation_matrix (CorrelationMatrix): The correlation matrix.
        """
        u_node = Context.active().u_node
        super().__init__([u_node])
        self.correlation = correlation_matrix

    def forward(self, ctx, u):
        """
        Transforms uncorrelated samples u to correlated samples z.

        Args:
            ctx (Context): The evaluation context.
            u (Tensor): Standard normal samples of shape [B, n].

        Returns:
            Tensor: Correlated samples z of shape [B, n].
        """
        L = self.correlation.get_L()
        return u @ L.T
