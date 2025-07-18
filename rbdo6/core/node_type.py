# ============================================================
# File        : node_type.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines enums for node types and differentiation modes
# ============================================================

from enum import Enum, auto


class NodeKind(Enum):
    """
    Enum specifying the role of a Node in the computation graph.

    Attributes:
        STANDARD: A normal operation node.
        U: A node that outputs the standard normal vector u.
        V: A node that outputs the design variable vector v.
    """
    STANDARD = auto()
    U = auto()
    V = auto()


class DerivativeMode(Enum):
    """
    Enum specifying how gradients are computed.

    Attributes:
        ANALYTIC: Supports autograd-based differentiation.
        NUMERIC: Uses numerical approximation (e.g., for black-box nodes).
    """
    ANALYTIC = auto()
    NUMERIC = auto()
