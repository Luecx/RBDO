# ============================================================
# File        : hvp_provider.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines an interface for Hessian-vector
#               product providers (HVP) to plug into nodes.
# ============================================================

class HVPProvider:
    """
    Base interface for Hessian-vector product (HVP) providers.

    Subclasses must implement the __call__ method to compute the
    product H @ vec, where H is the Hessian of the node's scalar
    output with respect to either `u` or `v`.

    This interface is compatible with batched inputs.
    """

    def __call__(self, node, vec, u, v, wrt="u"):
        """
        Computes the Hessian-vector product H @ vec, where H is the
        Hessian of the node output w.r.t. the specified variable.

        Args:
            node (Node): The node whose scalar output y = f(u, v) is used.
            vec (Tensor): The vector (or batch of vectors) to multiply with H.
            u (Tensor): Standard normal input tensor of shape [B, D_u].
            v (Tensor): Design variable tensor of shape [B, D_v].
            wrt (str): The variable to differentiate with respect to ("u" or "v").

        Returns:
            Tensor: Batched result of H @ vec for each sample in the batch.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("HVPProvider subclasses must implement __call__.")
