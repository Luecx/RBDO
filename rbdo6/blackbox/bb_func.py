# ============================================================
# File        : bb_func.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines a custom autograd function for black-box
#               operations with forward evaluation and finite
#               difference gradients for backward pass.
# ============================================================

import torch
from ..core import Context


class BlackBoxFunction(torch.autograd.Function):
    """
    Custom autograd Function for black-box operations in batched settings.

    Uses forward-mode function evaluation and backward-mode gradient estimation
    via finite differences. Assumes all inputs have shape [B, D] or [B].

    The function `fn(*inputs)` should return a tensor of shape [B].

    Note:
        This class is designed for use with unknown or non-differentiable Python functions,
        where autograd is not available. It falls back to finite-difference approximation
        for ∂y/∂x.
    """

    @staticmethod
    def forward(ctx, fn, *inputs):
        """
        Forward pass for the black-box function.

        Args:
            ctx: PyTorch context object for saving intermediate data.
            fn (callable): A Python function that takes batched inputs and returns [B]-shaped output.
            *inputs (torch.Tensor): Each of shape [B, D] or [B].

        Returns:
            torch.Tensor: Output tensor of shape [B].
        """
        ctx.fn = fn
        ctx.eps = 1e-2  # Finite difference epsilon
        ctx.save_for_backward(*inputs)
        return fn(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using forward finite differences to estimate gradients.

        Args:
            grad_output (torch.Tensor): Gradient from downstream (shape [B]).

        Returns:
            tuple: (None, ∂y/∂x1, ∂y/∂x2, ..., ∂y/∂xn)
        """
        Context.active().stats["blackbox_backward"] += 1

        inputs = ctx.saved_tensors
        fn  = ctx.fn
        eps = ctx.eps
        grads = []

        y0 = fn(*inputs)  # shape: [B]
        for i, x in enumerate(inputs):
            if x.ndim == 1:
                x = x.unsqueeze(0)  # [1, D]

            B, D = x.shape
            dx = torch.zeros_like(x)

            for j in range(D):
                x_pert = x.clone()
                x_pert[:, j] += eps

                x_inputs = list(inputs)
                x_inputs[i] = x_pert

                y1 = fn(*x_inputs)  # shape: [B]
                dx[:, j] = (y1 - y0) / eps

                Context.active().stats["blackbox_forward"] += 1

            grad_output_ = grad_output.view(B, 1)  # shape: [B, 1]
            grads.append(grad_output_ * dx)

        return (None, *grads)
