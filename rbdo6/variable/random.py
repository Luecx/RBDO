# ============================================================
# File        : random.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Base class for all random variables, providing
#               registration in the active Context and utilities
#               for resolving design-dependent parameters.
# ============================================================

import torch
from ..core import Context
from .design import DesignVariable


class RandomVariable:
    """
    Abstract base class for all random variables used in the RBDO framework.

    Each instance is automatically registered in the active Context to
    assign a unique ID for sampling and transformation.

    Subclasses must override the `sample(z_i)` method to define how standard
    normal inputs are mapped to physical space.
    """

    def __init__(self):
        """
        Registers the random variable with the active Context.
        """
        self._id = None
        Context.active().register_random(self)

    def sample(self, z_i):
        """
        Transforms standard normal samples to physical space.
        This must be implemented by subclasses.

        Args:
            z_i (torch.Tensor): Standard normal samples.

        Returns:
            torch.Tensor: Transformed physical-space values.
        """
        raise NotImplementedError("RandomVariable subclasses must implement sample(z_i).")

    @staticmethod
    def get_value(x):
        """
        Returns the tensor value of a DesignVariable or converts constants to tensors.

        Args:
            x (float | torch.Tensor | DesignVariable): Input parameter.

        Returns:
            torch.Tensor: Tensor representation of the value.
        """
        if isinstance(x, DesignVariable):
            return x.value()
        elif isinstance(x, (float, int)):
            return torch.tensor(x, dtype=torch.float32)
        return x
