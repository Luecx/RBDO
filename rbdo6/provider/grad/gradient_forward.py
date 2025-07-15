# ============================================================
# File        : gradient_forward.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements forward finite-difference gradients
#               for black-box functions in batched form.
# ============================================================

import torch
from .gradient_provider import GradientProvider


class ForwardDifference(GradientProvider):
    """
    Finite-difference gradient provider using forward differences.

    Approximates ∂y/∂x using the formula:
        (f(x + ε) - f(x)) / ε
    """

    def __init__(self, eps=1e-2):
        """
        Initializes the forward-difference provider.

        Args:
            eps (float): Perturbation size ε for the finite difference.
        """
        self.eps = eps

    def compute_gradients(self, fn, inputs, y0=None):
        """
        Computes ∂y/∂x using forward differences.

        Args:
            fn (callable): Function to differentiate.
            inputs (List[Tensor]): Batched input tensors.
            y0 (Tensor, optional): Cached fn(*inputs).

        Returns:
            List[Tensor]: List of gradients with shape [B, D_i] for each input i.
        """
        grads = []
        if y0 is None:
            y0 = fn(*inputs)

        for i, x in enumerate(inputs):
            B, D = x.shape if x.ndim == 2 else (x.shape[0], 1)
            x = x.view(B, D)
            dx = torch.zeros_like(x)

            for j in range(D):
                x_pert = x.clone()
                x_pert[:, j] += self.eps

                x_inputs = list(inputs)
                x_inputs[i] = x_pert

                y1 = fn(*x_inputs)
                dx[:, j] = (y1 - y0) / self.eps

            grads.append(dx)

        return grads
