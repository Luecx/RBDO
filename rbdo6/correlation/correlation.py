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

    Attributes:
        R (torch.Tensor): The correlation matrix [n × n].
        L (torch.Tensor): The Cholesky factor of R (computed via compile()).
        _dirty (bool): Tracks whether the matrix needs to be recompiled.
    """

    def __init__(self):
        """
        Initializes the correlation matrix based on Context.active().random.
        Raises an error if no random variables are registered.
        """
        self.random_vars = Context.active().random
        self.n = len(self.random_vars)

        if self.n == 0:
            raise RuntimeError("No random variables registered in context.")

        self.R = torch.eye(self.n)
        self.L = None
        self._dirty = True

    def set_correlation(self, r1, r2, rho: float):
        """
        Sets a correlation coefficient between two random variables.

        Args:
            r1 (RandomVariable): First variable.
            r2 (RandomVariable): Second variable.
            rho (float): Correlation coefficient to assign.
        """
        i, j = r1._id, r2._id
        self.R[i, j] = self.R[j, i] = rho
        self._dirty = True

    def compile(self) -> torch.Tensor:
        """
        Computes the Cholesky factor L of the correlation matrix R.

        Returns:
            torch.Tensor: The Cholesky factor L.
        """
        self.L = torch.linalg.cholesky(self.R).to(dtype=torch.float32)
        self._dirty = False
        return self.L

    def get_L(self) -> torch.Tensor:
        """
        Returns the Cholesky factor of the correlation matrix.

        Returns:
            torch.Tensor: Cholesky factor L [n × n].

        Raises:
            RuntimeError: If the matrix has not been compiled.
        """
        if self._dirty or self.L is None:
            raise RuntimeError("CorrelationMatrix has not been compiled. Call `.compile()` first.")
        return self.L
