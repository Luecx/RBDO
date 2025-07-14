# ============================================================
# File        : context.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Manages the global context for variable registration,
#               sampling inputs, and tracking execution statistics.
#               Provides access to design and random variables and
#               ensures proper gradient handling for optimization.
# ============================================================

import torch


class Context:
    """
    Global context for managing design and random variables in RBDO.

    This context tracks:
    - Registered design and random variables
    - Input values for u (standard normal) and v (design variables)
    - Execution statistics for forward and backward passes

    Only one context can be active at a time, and it must be accessed using a `with` statement.
    """

    _active = None  # Static reference to the currently active context

    def __init__(self):
        """
        Initializes an empty context with variable lists and statistics.
        """
        self.design = []  # List of registered design variables
        self.random = []  # List of registered random variables
        self.u = None     # Input vector in standard normal space
        self.v = None     # Input vector for design variables

        self.stats = {
            "forward_calls": 0,
            "blackbox_forward": 0,
            "blackbox_backward": 0
        }

    def __enter__(self):
        """
        Enters the context, making it the globally active context.
        """
        Context._active = self
        return self

    def __exit__(self, *args):
        """
        Exits the context, deactivating it.
        """
        Context._active = None

    def register_design(self, var):
        """
        Registers a design variable with the context.

        Args:
            var: A DesignVariable instance.
        """
        var._id = len(self.design)
        self.design.append(var)

    def register_random(self, var):
        """
        Registers a random variable with the context.

        Args:
            var: A RandomVariable instance.
        """
        var._id = len(self.random)
        self.random.append(var)

    def set_inputs(self, u, v):
        """
        Sets the batched input values for u (standard normal) and v (design space).

        Both u and v must be 2D tensors with gradients enabled for backpropagation.

        Args:
            u (torch.Tensor): Standard normal inputs of shape [B, n_u] or [n_u].
            v (torch.Tensor): Design variable inputs of shape [B, n_v] or [n_v].
        """
        if u.ndim == 1:
            u = u.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if not u.requires_grad:
            u = u.clone().detach().requires_grad_(True)
        if not v.requires_grad:
            v = v.clone().detach().requires_grad_(True)

        self.u = u
        self.v = v

    @staticmethod
    def active():
        """
        Returns the currently active context.

        Returns:
            Context: The currently active context.

        Raises:
            RuntimeError: If no context is currently active.
        """
        if Context._active is None:
            raise RuntimeError("No active context.")
        return Context._active
