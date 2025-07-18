# ============================================================
# File        : node_input.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines input nodes for standard normal (U)
#               and design variable (V) vectors.
# ============================================================

from .node import *
from .node_type import NodeKind


class UNode(Node):
    """
    Node representing the standard normal input vector (u).
    Used as the source of randomness in the graph.
    """

    def __init__(self):
        super().__init__(kind=NodeKind.U)

    def forward(self, ctx, u: torch.Tensor) -> torch.Tensor:
        """
        Forwards the standard normal input directly.

        Args:
            ctx (Context): Active computation context.
            u (torch.Tensor): Standard normal input [B, n_u].

        Returns:
            torch.Tensor: Same as input.
        """
        return u


class VNode(Node):
    """
    Node representing the design variable input vector (v).
    Used to supply the deterministic design parameters to the graph.
    """

    def __init__(self):
        super().__init__(kind=NodeKind.V)

    def forward(self, ctx, v: torch.Tensor) -> torch.Tensor:
        """
        Forwards the design variable input directly.

        Args:
            ctx (Context): Active computation context.
            v (torch.Tensor): Design variable input [B, n_v].

        Returns:
            torch.Tensor: Same as input.
        """
        return v
