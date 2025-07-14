from .context import Context
from torch.autograd.functional import hvp
import torch


class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self._uses_numerical = any(getattr(inp, "_uses_numerical", False) for inp in self.inputs)
        self.hvp_provider = None
        self.hesse_provider = None

    def add_hvp_provider(self, provider):
        self.hvp_provider = provider

    def add_hesse_provider(self, provider):
        self.hesse_provider = provider

    def hvp_u(self, v):
        if self.hvp_provider is None:
            raise RuntimeError("No HVP provider set.")
        return self.hvp_provider(self, v)

    def hesse_u(self):
        if self.hesse_provider is None:
            raise RuntimeError("No Hesse provider set.")
        return self.hesse_provider(self)

    def forward(self, ctx, *args):
        raise NotImplementedError

    def call(self, u=None, v=None, grad=False, gradgrad_u=False, gradgrad_v=False, hvp_u=None, hvp_v=None):
        ctx = Context.active()

        if u is not None and v is not None:
            ctx.set_inputs(u, v)

        values = []
        for inp in self.inputs:
            if isinstance(inp, Node):
                values.append(inp.call(u=ctx.u, v=ctx.v, grad=False)["out"])
            else:
                values.append(inp)

        ctx.stats["forward_calls"] += 1
        out = self.forward(ctx, *values)
        result = {"out": out}

        if grad and out.requires_grad:
            if u is not None:
                if not u.requires_grad:
                    u.requires_grad_(True)
                u.retain_grad()

            if v is not None:
                if not v.requires_grad:
                    v.requires_grad_(True)
                v.retain_grad()

            if ctx.u.grad is not None:
                ctx.u.grad.zero_()
            if ctx.v.grad is not None:
                ctx.v.grad.zero_()

            out.backward(torch.ones_like(out), retain_graph=True)

            result["grad_u"] = u.grad.clone() if u is not None and u.grad is not None else None
            result["grad_v"] = v.grad.clone() if v is not None and v.grad is not None else None

        if gradgrad_u and not self._uses_numerical:
            hess_u_list = []
            for i in range(ctx.u.shape[0]):
                hess_u = torch.autograd.functional.hessian(
                    lambda uu: self.call(u=uu.unsqueeze(0), v=ctx.v[i].unsqueeze(0), grad=False)["out"].squeeze(0),
                    ctx.u[i], create_graph=True)
                hess_u_list.append(hess_u)
            result["hess_u"] = torch.stack(hess_u_list)

        if gradgrad_v and not self._uses_numerical:
            hess_v_list = []
            for i in range(ctx.v.shape[0]):
                hess_v = torch.autograd.functional.hessian(
                    lambda vv: self.call(u=ctx.u[i].unsqueeze(0), v=vv.unsqueeze(0), grad=False)["out"].squeeze(0),
                    ctx.v[i], create_graph=True)
                hess_v_list.append(hess_v)
            result["hess_v"] = torch.stack(hess_v_list)

        if hvp_u is not None:
            if self.hvp_provider is not None:
                result["hvp_u"] = self.hvp_provider(self, hvp_u)
            elif not self._uses_numerical:
                hvps = []
                for i in range(ctx.u.shape[0]):
                    _, hvp_val = torch.autograd.functional.hvp(
                        lambda uu: self.call(u=uu.unsqueeze(0), v=ctx.v[i].unsqueeze(0), grad=False)["out"].squeeze(0),
                        ctx.u[i], hvp_u[i])
                    hvps.append(hvp_val)
                result["hvp_u"] = torch.stack(hvps)

        if hvp_v is not None:
            if self.hvp_provider is not None:
                result["hvp_v"] = self.hvp_provider(self, hvp_v)
            elif not self._uses_numerical:
                hvps = []
                for i in range(ctx.v.shape[0]):
                    _, hvp_val = torch.autograd.functional.hvp(
                        lambda vv: self.call(u=ctx.u[i].unsqueeze(0), v=vv.unsqueeze(0), grad=False)["out"].squeeze(0),
                        ctx.v[i], hvp_v[i])
                    hvps.append(hvp_val)
                result["hvp_v"] = torch.stack(hvps)

        result["stats"] = dict(ctx.stats)
        return result