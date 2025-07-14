# ============================================================
# File        : index_node.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements a node that extracts a specific
#               column (feature) from a batched input tensor.
# ============================================================

from .node import *


class IndexNode(Node):
    """
    Node that selects a single column (feature) from a batched input tensor.

    Useful for extracting individual components from a multi-dimensional output.

    Attributes:
        index (int): Index of the column to extract from the input tensor.
    """

    def __init__(self, index, source_node):
        """
        Initializes an IndexNode that selects the i-th column from its input.

        Args:
            index (int): Index of the column to extract.
            source_node (Node): The upstream node providing a [B, N] output.
        """
        super().__init__([source_node])
        self.index = index

    def forward(self, ctx, x):
        """
        Forwards the selected column from the input.

        Args:
            x (torch.Tensor): Batched input tensor of shape [B, N].

        Returns:
            torch.Tensor: Extracted column vector of shape [B].
        """
        return x[:, self.index]
