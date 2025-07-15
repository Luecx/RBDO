# ============================================================
# File        : gradient_central.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements central finite-difference gradients
#               for black-box functions in batched form.
# ============================================================

import torch
from .gradient_provider import GradientProvider


class CentralDifference(GradientProvider):
    """
    Finite-difference gradient provider using central differences.

    Approximates ∂y/∂x using the formula:
        (f(x + ε) - f(x - ε)) / (2ε)
    """

    def __init__(self, eps=1e-4):
        """
        Initializes the central-difference provider.

        Args:
            eps (float): Perturbation size ε for the finite difference.
        """
        self.eps = eps

    def compute_gradients(self, fn, inputs, y0=None):
        """
        Computes ∂y/∂x using central differences.

        Args:
            fn (callable): Function to differentiate.
            inputs (List[Tensor]): Batched input tensors.
            y0 (Tensor, optional): Cached fn(*inputs), unused here.

        Returns:
            List[Tensor]: List of gradients with shape [B, D_i] for each input i.
        """
        grads = []

        for i, x in enumerate(inputs):
            B, D = x.shape if x.ndim == 2 else (x.shape[0], 1)
            x = x.view(B, D)
            dx = torch.zeros_like(x)

            for j in range(D):
                x_p = x.clone()
                x_m = x.clone()
                x_p[:, j] += self.eps
                x_m[:, j] -= self.eps

                x_inputs_p = list(inputs)
                x_inputs_m = list(inputs)
                x_inputs_p[i] = x_p
                x_inputs_m[i] = x_m

                y1 = fn(*x_inputs_p)
                y2 = fn(*x_inputs_m)

                dx[:, j] = (y1 - y2) / (2 * self.eps)

            grads.append(dx)

        return grads
