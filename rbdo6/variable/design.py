# ============================================================
# File        : design.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines the DesignVariable class that registers
#               with the active Context and enables access to
#               batched design values via index-based lookup.
# ============================================================

from ..core import Context


class DesignVariable:
    """
    Represents a design variable in the RBDO computational graph.

    Upon construction, the variable is automatically registered
    with the current Context. It retrieves its current value via
    indexing into the design variable tensor `v` stored in Context.

    Attributes:
        name (str): The name of the design variable (for debugging/logging).
        _id (int): Index in the Context's design variable list (set on registration).
        _value (float): Initial value (only used for logging or potential defaults).
    """

    def __init__(self, name, value):
        """
        Initializes and registers the design variable with the current context.

        Args:
            name (str): A unique identifier for the design variable.
            value (float): The initial (default) value of the variable.
        """
        self.name = name
        self._id = None
        self._value = value
        Context.active().register_design(self)

    def value(self):
        """
        Returns the batched values of this design variable.

        Returns:
            torch.Tensor: A 1D tensor of shape [B], where B is the batch size.
        """
        return Context.active().v[:, self._id]
