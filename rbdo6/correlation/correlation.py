# ============================================================
# File        : correlation.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Manages a correlation matrix between random variables.
# ============================================================

import torch
from ..core import Context


class CorrelationMatrix:
    """
    Represents the correlation structure between registered random variables.

    Uses Cholesky decomposition to enable the Nataf transformation:
        z = u @ Lᵀ, where L is the Cholesky factor of R.
    """

    def __init__(self):
        """
        Initializes the correlation matrix based on Context.active().random.
        """
        self.random_vars = Context.active().random
        self.n = len(self.random_vars)

        if self.n == 0:
            raise RuntimeError("No random variables registered in context.")

        self.R = torch.eye(self.n)
        self.L = None
        self._dirty = True

    def set_correlation(self, r1, r2, rho):
        """
        Sets a correlation coefficient between two random variables.

        Args:
            r1 (RandomVariable): First variable.
            r2 (RandomVariable): Second variable.
            rho (float): Correlation coefficient between r1 and r2.
        """
        i, j = r1._id, r2._id
        self.R[i, j] = self.R[j, i] = rho
        self._dirty = True

    def compile(self):
        """
        Computes the Cholesky factor L of the correlation matrix.
        """
        self.L = torch.linalg.cholesky(self.R).to(dtype=torch.float32)
        self._dirty = False

    def get_L(self):
        """
        Returns the Cholesky factor of the correlation matrix.

        Returns:
            Tensor: The Cholesky factor L [n × n].

        Raises:
            RuntimeError: If compile() has not been called.
        """
        if self._dirty:
            raise RuntimeError("CorrelationMatrix not compiled.")
        return self.L
