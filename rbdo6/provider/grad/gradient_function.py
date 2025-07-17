# ============================================================
# File        : grad_function.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Wraps a node forward pass with custom gradient
#               computation using a GradientProvider.
# ============================================================

import torch

class CustomGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_provider, fn, *inputs):
        ctx.grad_provider = grad_provider
        ctx.fn = fn
        ctx.save_for_backward(*inputs)
        with torch.no_grad():
            return fn(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        grads = ctx.grad_provider.compute_gradients(ctx.fn, inputs)
        return (None, None, *[g * grad_output for g in grads])
