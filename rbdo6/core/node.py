# ============================================================
# File        : node.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Defines the base class for all computational
#               nodes in the computation graph. Handles forward
#               evaluation, gradient and Hessian computation,
#               as well as Hessian-vector products.
# ============================================================

from .context import Context
from .node_type import *
from ..provider.grad import GradientProvider, CustomGradFunction
from ..provider.hvp import HVPProvider
from torch.autograd.functional import hvp
import torch

class Node:
    def __init__(self, inputs=None, *, kind=NodeKind.STANDARD,
                 grad_provider=None,
                 hesse_provider=None,
                 hvp_provider=None):

        self.inputs = inputs if inputs else []
        self.kind = kind

        self.grad_provider  = grad_provider
        self.hesse_provider = hesse_provider
        self.hvp_provider   = hvp_provider

        if grad_provider:
            self.grad_mode = DerivativeMode.NUMERIC
        elif any(isinstance(inp, Node) and inp.grad_mode == DerivativeMode.NUMERIC for inp in self.inputs):
            self.grad_mode = DerivativeMode.NUMERIC
        else:
            self.grad_mode = DerivativeMode.ANALYTIC

        self._cache = None
        self._cache_key = None

    def forward(self, ctx, *args):
        raise NotImplementedError

    def get_inputs(self, u, v, _call_id):
        values = []
        for inp in self.inputs:
            if isinstance(inp, Node):
                if inp.kind == NodeKind.U:
                    values.append(u)
                elif inp.kind == NodeKind.V:
                    values.append(v)
                else:
                    values.append(inp.call(u=u, v=v, _call_id=_call_id)["out"])
            else:
                values.append(inp)
        return values

    def call(self, u=None, v=None, grad=False, gradgrad_u=False, gradgrad_v=False, hvp_u=None, hvp_v=None, _call_id=None):
        root = _call_id is None
        if root:
            u, v = self._ensure_batched(u, v)

        ctx = Context.active()
        result = {}

        if _call_id is None:
            _call_id = ctx._next_call_id()

        if _call_id != self._cache_key:
            inputs = self.get_inputs(u, v, _call_id)
            ctx.stats["forward_calls"] += 1
            if self.grad_provider is not None:
                out = CustomGradFunction.apply(self.grad_provider, lambda *x: self.forward(ctx, *x), *inputs)
            else:
                out = self.forward(ctx, *inputs)
        else:
            out = self._cache

        result["out"] = out
        self._cache_key = _call_id
        self._cache = out

        # if not at root, we can exit
        if not root:
            return result

        # Gradients
        if grad:
            if out.requires_grad:
                if u is not None and not u.requires_grad:
                    u.requires_grad_(True)
                    u.retain_grad()
                if v is not None and not v.requires_grad:
                    v.requires_grad_(True)
                    v.retain_grad()
                if u.grad is not None:
                    u.grad.zero_()
                if v.grad is not None:
                    v.grad.zero_()
                out.backward(torch.ones_like(out), retain_graph=True)
                result["grad_u"] = u.grad.clone() if u is not None and u.grad is not None else None
                result["grad_v"] = v.grad.clone() if v is not None and v.grad is not None else None
            else:
                raise RuntimeError("Cannot compute gradients due to disconnection")
        # Second-order
        if gradgrad_u:
            result["hess_u"] = self._compute_hessian_u(u, v)
        if gradgrad_v:
            result["hess_v"] = self._compute_hessian_v(u, v)
        if hvp_u is not None:
            result["hvp_u"] = self._compute_hvp_u(u, v, hvp_u)
        if hvp_v is not None:
            result["hvp_v"] = self._compute_hvp_v(u, v, hvp_v)

        return result

    def _ensure_batched(self, u, v):
        if u is not None and u.ndim == 1:
            u = u.unsqueeze(0)
        if v is not None and v.ndim == 1:
            v = v.unsqueeze(0)
        return u, v

    def _compute_hessian_u(self, u, v):
        if self.hesse_provider is not None:
            return self.hesse_provider(self, u, v, wrt="u")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute Hessian for numerically differentiated node without a hesse_provider.")
        hess_u_list = []
        for i in range(u.shape[0]):
            u_i = u[i].detach().requires_grad_()
            v_i = v[i].detach()
            hess_u = torch.autograd.functional.hessian(
                lambda uu: self.call(u=uu.unsqueeze(0), v=v_i.unsqueeze(0), grad=False)["out"].squeeze(0),
                u_i, create_graph=True)
            hess_u_list.append(hess_u)
        return torch.stack(hess_u_list)

    def _compute_hessian_v(self, u, v):
        if self.hesse_provider is not None:
            return self.hesse_provider(self, u, v, wrt="v")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute Hessian for numerically differentiated node without a hesse_provider.")
        hess_v_list = []
        for i in range(v.shape[0]):
            u_i = u[i].detach()
            v_i = v[i].detach().requires_grad_()
            hess_v = torch.autograd.functional.hessian(
                lambda vv: self.call(u=u_i.unsqueeze(0), v=vv.unsqueeze(0), grad=False)["out"].squeeze(0),
                v_i, create_graph=True)
            hess_v_list.append(hess_v)
        return torch.stack(hess_v_list)

    def _compute_hvp_u(self, u, v, hvp_vecs):
        if self.hvp_provider is not None:
            return self.hvp_provider(self, hvp_vecs, u, v, wrt="u")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute HVP for numerically differentiated node without a hvp_provider.")
        hvps = []
        for i in range(u.shape[0]):
            u_i = u[i].detach().requires_grad_()
            v_i = v[i].detach()
            _, hvp_val = hvp(
                lambda uu: self.call(u=uu.unsqueeze(0), v=v_i.unsqueeze(0), grad=False)["out"].squeeze(0),
                u_i, hvp_vecs[i])
            hvps.append(hvp_val)
        return torch.stack(hvps)

    def _compute_hvp_v(self, u, v, hvp_vecs):
        if self.hvp_provider is not None:
            return self.hvp_provider(self, hvp_vecs, u, v, wrt="v")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute HVP for numerically differentiated node without a hvp_provider.")
        hvps = []
        for i in range(v.shape[0]):
            u_i = u[i].detach()
            v_i = v[i].detach().requires_grad_()
            _, hvp_val = hvp(
                lambda vv: self.call(u=u_i.unsqueeze(0), v=vv.unsqueeze(0), grad=False)["out"].squeeze(0),
                v_i, hvp_vecs[i])
            hvps.append(hvp_val)
        return torch.stack(hvps)

    def set_grad_provider(self, provider):
        if not isinstance(provider, GradientProvider):
            raise TypeError("grad_provider must be an instance of GradientProvider.")
        self.grad_provider = provider
        self.grad_mode = DerivativeMode.NUMERIC

    def set_hvp_provider(self, provider):
        if not isinstance(provider, HVPProvider):
            raise TypeError("hvp_provider must be an instance of HVPProvider.")
        self.hvp_provider = provider

    def set_hesse_provider(self, func):
        self.hesse_provider = func
