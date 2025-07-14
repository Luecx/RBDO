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
from torch.autograd.functional import hvp
import torch


class Node:
    """
    Base class for all computation graph nodes.

    A Node represents a differentiable operation that can be part of a
    computational graph, enabling forward evaluation, backward gradients,
    Hessians, and Hessian-vector products (HVPs). Supports batching over inputs.

    Attributes:
        inputs (list): Upstream nodes or constant values.
        _uses_numerical (bool): Whether any input node uses numerical derivatives.
        hvp_provider (callable): Optional user-defined Hessian-vector product.
        hesse_provider (callable): Optional user-defined full Hessian provider.
    """

    def __init__(self, inputs=None):
        """
        Initializes a Node and inspects its inputs.

        Args:
            inputs (list, optional): Upstream nodes or constants.
        """
        self.inputs = inputs if inputs else []
        self._uses_numerical = any(getattr(inp, "_uses_numerical", False) for inp in self.inputs)
        self.hvp_provider = None
        self.hesse_provider = None

    def add_hvp_provider(self, provider):
        """
        Registers a custom Hessian-vector product provider.

        Args:
            provider (callable): Function (self, v) → hvp result.
        """
        self.hvp_provider = provider

    def add_hesse_provider(self, provider):
        """
        Registers a custom full Hessian provider.

        Args:
            provider (callable): Function (self) → Hessian.
        """
        self.hesse_provider = provider

    def hvp_u(self, v):
        """
        Computes HVP with respect to u using the registered provider.

        Args:
            v (torch.Tensor): Vector to multiply with Hessian.

        Returns:
            torch.Tensor: Hessian-vector product result.
        """
        if self.hvp_provider is None:
            raise RuntimeError("No HVP provider set.")
        return self.hvp_provider(self, v)

    def hesse_u(self):
        """
        Computes full Hessian with respect to u using the registered provider.

        Returns:
            torch.Tensor: Full Hessian matrix.
        """
        if self.hesse_provider is None:
            raise RuntimeError("No Hesse provider set.")
        return self.hesse_provider(self)

    def forward(self, ctx, *args):
        """
        Must be implemented by subclasses to define the computation.

        Args:
            ctx (Context): Current active context.
            *args: Input values from upstream nodes.

        Returns:
            torch.Tensor: Output tensor of shape [B].
        """
        raise NotImplementedError

    def call(self, u=None, v=None, grad=False, gradgrad_u=False, gradgrad_v=False, hvp_u=None, hvp_v=None):
        """
        Executes a forward pass through the node with optional derivatives.

        Args:
            u (torch.Tensor, optional): Standard normal input (shape [B, n_u]).
            v (torch.Tensor, optional): Design variable input (shape [B, n_v]).
            grad (bool): Whether to compute gradients w.r.t. u and v.
            gradgrad_u (bool): Whether to compute full Hessian w.r.t. u.
            gradgrad_v (bool): Whether to compute full Hessian w.r.t. v.
            hvp_u (torch.Tensor, optional): Vector(s) for HVP in u-space.
            hvp_v (torch.Tensor, optional): Vector(s) for HVP in v-space.

        Returns:
            dict: {
                "out": forward output,
                "grad_u": ∇u if requested,
                "grad_v": ∇v if requested,
                "hess_u": H_u if requested,
                "hess_v": H_v if requested,
                "hvp_u": H_u @ v if requested,
                "hvp_v": H_v @ v if requested,
                "stats": dict of context stats
            }
        """
        ctx = Context.active()

        if u is not None and v is not None:
            ctx.set_inputs(u, v)

        # Evaluate input nodes recursively
        values = []
        for inp in self.inputs:
            if isinstance(inp, Node):
                values.append(inp.call(u=ctx.u, v=ctx.v, grad=False)["out"])
            else:
                values.append(inp)

        ctx.stats["forward_calls"] += 1
        out = self.forward(ctx, *values)
        result = {"out": out}

        # First-order gradients
        if grad and out.requires_grad:
            if u is not None and not u.requires_grad:
                u.requires_grad_(True)
                u.retain_grad()
            if v is not None and not v.requires_grad:
                v.requires_grad_(True)
                v.retain_grad()

            if ctx.u.grad is not None:
                ctx.u.grad.zero_()
            if ctx.v.grad is not None:
                ctx.v.grad.zero_()

            out.backward(torch.ones_like(out), retain_graph=True)

            result["grad_u"] = u.grad.clone() if u is not None and u.grad is not None else None
            result["grad_v"] = v.grad.clone() if v is not None and v.grad is not None else None

        # Second-order derivatives (full Hessians)
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

        # HVP in u-space
        if hvp_u is not None:
            if self.hvp_provider is not None:
                result["hvp_u"] = self.hvp_provider(self, hvp_u)
            elif not self._uses_numerical:
                hvps = []
                for i in range(ctx.u.shape[0]):
                    _, hvp_val = hvp(
                        lambda uu: self.call(u=uu.unsqueeze(0), v=ctx.v[i].unsqueeze(0), grad=False)["out"].squeeze(0),
                        ctx.u[i], hvp_u[i])
                    hvps.append(hvp_val)
                result["hvp_u"] = torch.stack(hvps)

        # HVP in v-space
        if hvp_v is not None:
            if self.hvp_provider is not None:
                result["hvp_v"] = self.hvp_provider(self, hvp_v)
            elif not self._uses_numerical:
                hvps = []
                for i in range(ctx.v.shape[0]):
                    _, hvp_val = hvp(
                        lambda vv: self.call(u=ctx.u[i].unsqueeze(0), v=vv.unsqueeze(0), grad=False)["out"].squeeze(0),
                        ctx.v[i], hvp_v[i])
                    hvps.append(hvp_val)
                result["hvp_v"] = torch.stack(hvps)

        result["stats"] = dict(ctx.stats)
        return result
