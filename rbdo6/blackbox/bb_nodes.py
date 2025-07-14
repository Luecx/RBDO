# ============================================================
# File        : bb_nodes.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines a computation node that wraps a black-box
#               function and delegates differentiation to a
#               finite-difference-based custom autograd function.
# ============================================================

from .bb_func import BlackBoxFunction
from ..core import Node


class BlackBoxNode(Node):
    """
    Node for wrapping a black-box function using finite-difference gradients.

    This node delegates both forward evaluation and gradient computation
    to a BlackBoxFunction, which uses PyTorch’s custom autograd support.

    Attributes:
        fn (callable): The black-box function to evaluate. Must return shape [B].
        _uses_numerical (bool): Flag indicating use of numerical derivatives.
    """

    def __init__(self, fn, input_nodes):
        """
        Initializes a BlackBoxNode.

        Args:
            fn (callable): A function that takes batched input tensors and returns [B]-shaped output.
            input_nodes (list[Node]): List of input nodes for the function.
        """
        super().__init__(input_nodes)
        self.fn = fn
        self._uses_numerical = True  # Disables exact second-order derivatives

    def forward(self, ctx, *args):
        """
        Performs the forward pass through the black-box function.

        Args:
            ctx (Context): Active computation context.
            *args: Inputs from upstream nodes (batched).

        Returns:
            torch.Tensor: Output tensor of shape [B].
        """
        return BlackBoxFunction.apply(self.fn, *args)
