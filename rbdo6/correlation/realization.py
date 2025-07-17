# ============================================================
# File        : realization.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Transforms correlated samples z into physical samples x
#               by passing them through each registered random variable.
# ============================================================

import torch
from ..core import Node, IndexNode, Context


class Realization(Node):
    """
    Transforms correlated samples z into physical samples x using
    the random variables registered in the active context.

    For each random variable:
        x_i = rv.sample(z_i, v)
    """

    def __init__(self, z_node):
        """
        Initializes the Realization node.

        Args:
            z_node (Node): Node producing correlated samples z.
        """
        v_node = Context.active().v_node
        super().__init__([z_node, v_node])

    def forward(self, ctx, z, v):
        """
        Evaluates all random variable transformations at z.

        Args:
            ctx (Context): The current evaluation context.
            z (Tensor): Correlated standard normal samples of shape [B, N].
            v (Tensor): Design variable samples of shape [B, n_v].

        Returns:
            Tensor: Physical samples x of shape [B, N].
        """

        random_vars = ctx.random
        x_list = []

        for i, rv in enumerate(random_vars):
            z_i = z[:, i]                # [B]
            x_i = rv.sample(z_i, v=v)    # [B]
            x_list.append(x_i.unsqueeze(1))

        return torch.cat(x_list, dim=1)  # [B, N]

    def __getitem__(self, rv):
        """
        Enables indexing into the output x by random variable.

        Args:
            rv (RandomVariable): The random variable to index.

        Returns:
            IndexNode: Node that extracts x_i from x.
        """
        return IndexNode(rv._id, self)
