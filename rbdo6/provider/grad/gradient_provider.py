# ============================================================
# File        : gradient_provider.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines an abstract interface for gradient
#               providers, allowing different differentiation
#               schemes (e.g., forward, central) to be plugged
#               into black-box nodes.
# ============================================================


class GradientProvider:
    """
    Abstract base class for black-box gradient providers.

    Subclasses must implement the `compute_gradients` method
    which approximates ∂y/∂x for all input tensors using a
    custom finite-difference or other scheme.
    """

    def compute_gradients(self, fn, inputs, y0=None):
        """
        Computes ∂y/∂x using a custom gradient approximation method.

        Args:
            fn (callable): The function whose gradient is being computed.
            inputs (List[Tensor]): Input tensors of shape [B, D] or [B].
            y0 (Tensor, optional): Cached output of fn(*inputs). Reused if available.

        Returns:
            List[Tensor]: Approximated gradients ∂y/∂xi for each input tensor.
        """
        raise NotImplementedError("GradientProvider requires implementation.")
