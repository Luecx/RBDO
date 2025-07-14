import torch
from ..core import Context

class HVP_FDiff_Provider:
    def __init__(self, eps=1e-2):
        self.eps = eps

    def __call__(self, node, v):
        ctx = Context.active()
        u_base = ctx.u.detach().clone()
        v = v.detach()

        def func(u_):
            ctx.set_inputs(u_.requires_grad_(True), ctx.v)
            return node.call(grad=True)["out"]

        u_plus = (u_base + self.eps * v).requires_grad_(True)
        ctx.set_inputs(u_plus, ctx.v)
        g1 = torch.autograd.grad(func(u_plus), u_plus, retain_graph=False)[0]

        u_base.requires_grad_(True)
        ctx.set_inputs(u_base, ctx.v)
        g0 = torch.autograd.grad(func(u_base), u_base, retain_graph=False)[0]

        return (g1 - g0) / self.eps