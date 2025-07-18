# ============================================================
# File        : context.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Manages the global context for variable registration,
#               sampling inputs, and tracking execution statistics.
# ============================================================

import torch


class Context:
    """
    Global context manager for RBDO computations.

    This context tracks:
    - Registered design and random variables
    - Execution statistics for forward and backward passes
    - Input nodes for u (standard normal) and v (design variables)
    - A call ID counter for dependency tracking

    Use this context via a `with` statement to ensure correct scoping.
    """

    _active = None  # Static reference to the currently active context

    def __init__(self):
        """Initializes a new context with variable lists and tracking."""
        self.design = []  # List of registered DesignVariable instances
        self.random = []  # List of registered RandomVariable instances

        self.stats = {
            "forward_calls": 0,
            "blackbox_forward": 0,
            "blackbox_backward": 0
        }

        # Input nodes for computational graph
        from .node_input import UNode, VNode
        self.u_node = UNode()
        self.v_node = VNode()

        # Unique call ID counter for graph dependency tracking
        self.call_id = 0

    def _next_call_id(self) -> int:
        """
        Increments and returns the next unique call ID.

        Returns:
            int: The next call ID.
        """
        self.call_id += 1
        return self.call_id

    def __enter__(self) -> "Context":
        """
        Enters the context, making it the globally active one.

        Returns:
            Context: The active context instance.
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
        Registers a design variable with this context.

        Args:
            var: A DesignVariable instance to register.
        """
        var._id = len(self.design)
        self.design.append(var)

    def register_random(self, var):
        """
        Registers a random variable with this context.

        Args:
            var: A RandomVariable instance to register.
        """
        var._id = len(self.random)
        self.random.append(var)

    @staticmethod
    def active() -> "Context":
        """
        Returns the currently active context.

        Returns:
            Context: The active context instance.

        Raises:
            RuntimeError: If no context is currently active.
        """
        if Context._active is None:
            raise RuntimeError("No active context.")
        return Context._active
