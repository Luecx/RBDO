
import torch
from ..core import Context

class BlackBoxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *inputs):
        ctx.fn = fn
        ctx.eps = 1e-2
        ctx.save_for_backward(*inputs)
        return fn(*inputs)  # all inputs have shape [B]

    @staticmethod
    def backward(ctx, grad_output):
        Context.active().stats["blackbox_backward"] += 1  # ✅ Zähler für Backward-Aufrufe

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
                y1 = fn(*x_inputs)  # [B]
                dx[:, j] = (y1 - y0) / eps

                Context.active().stats["blackbox_forward"] += 1

            grad_output_ = grad_output.view(B, 1)
            grads.append(grad_output_ * dx)

        return (None, *grads)