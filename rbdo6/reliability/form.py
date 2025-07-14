# ============================================================
# File        : form.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the First-Order Reliability Method (FORM)
#               using the HL-RF algorithm with autograd support.
# ============================================================

import math
import torch
from torch.special import erf
from ..core import Node, Context


class FORMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, form_node, g_node, u, v, max_iter, tol, eta):
        with torch.enable_grad():
            B, n_u = u.shape
            ctx.g_node = g_node
            ctx.u = u
            ctx.v = v
            ctx.max_iter = max_iter
            ctx.tol = tol
            ctx.eta = eta
            ctx.form_node = form_node

            ctx.u_star = torch.zeros_like(u)
            ctx.beta = torch.zeros(B, device=u.device, dtype=u.dtype)
            ctx.du = []
            ctx.dv = []

            pf_out = torch.zeros(B, device=u.device, dtype=u.dtype)

            for b in range(B):
                u_b = torch.zeros(n_u, device=u.device, dtype=u.dtype)
                v_b = v[b]

                # Evaluate g at origin to determine sign of β
                g0_tmp = torch.zeros(1, n_u, device=u.device, dtype=u.dtype).requires_grad_(True)
                v_tmp = v_b.unsqueeze(0).clone().detach().requires_grad_(True)
                Context.active().set_inputs(g0_tmp, v_tmp)
                res0 = g_node.call(u=g0_tmp, v=v_tmp, grad=False)
                sign = -1.0 if res0["out"].item() < 0 else 1.0

                # HLRF Iteration
                for _ in range(max_iter):
                    u_tmp = u_b.unsqueeze(0).clone().detach().requires_grad_(True)
                    v_tmp = v_b.unsqueeze(0).clone().detach().requires_grad_(True)

                    res = g_node.call(u=u_tmp, v=v_tmp, grad=True)
                    g_val = res["out"].item()
                    du = res["grad_u"].squeeze(0)
                    norm2 = du.dot(du).item()

                    if abs(g_val) < tol or norm2 == 0.0:
                        break

                    lam = (du @ u_b - g_val) / norm2
                    u_new = u_b + eta * (lam * du - u_b)
                    if (u_new - u_b).norm().item() < tol:
                        u_b = u_new
                        break
                    u_b = u_new

                # Final call to get gradients at u*
                u_tmp = u_b.unsqueeze(0).clone().detach().requires_grad_(True)
                v_tmp = v_b.unsqueeze(0).clone().detach().requires_grad_(True)
                Context.active().set_inputs(u_tmp, v_tmp)

                res = g_node.call(u=u_tmp, v=v_tmp, grad=True)
                du = res["grad_u"].squeeze(0)
                dv = res["grad_v"].squeeze(0)

                β = sign * u_b.norm()
                pf_val = 0.5 * (1.0 - erf(β / math.sqrt(2.0)))

                pf_out[b] = pf_val
                ctx.u_star[b] = u_b
                ctx.beta[b] = β
                ctx.du.append(du)
                ctx.dv.append(dv)

            # Store result on the calling FORM instance
            form_node._last_beta = ctx.beta.detach().clone()

            return pf_out

    @staticmethod
    def backward(ctx, grad_out):
        u, v = ctx.u, ctx.v
        B, n_u = u.shape
        n_v = v.shape[1]
        grad_u = torch.zeros_like(u)
        grad_v = torch.zeros_like(v)

        for b in range(B):
            du = ctx.du[b]
            dv = ctx.dv[b]
            β = ctx.beta[b]
            norm_du = du.norm().item()

            if norm_du == 0.0:
                dpf_dg = 0.0
            else:
                dpf_dg = -math.exp(-β.item() ** 2 / 2.0) / math.sqrt(2.0 * math.pi) / norm_du

            grad_u[b] = dpf_dg * du * grad_out[b]
            grad_v[b] = dpf_dg * dv * grad_out[b]

        return None, None, grad_u, grad_v, None, None, None


class FORM(Node):
    """
    First-Order Reliability Method (FORM) node using the HL-RF algorithm.

    Computes the probability of failure Pf ≈ Φ(-β), where β is the reliability index.
    """

    def __init__(self, g_node, max_iter=50, tol=1e-6, eta=1.0):
        """
        Initializes the FORM node.

        Args:
            g_node (Node): The limit state function node.
            max_iter (int): Maximum number of HL-RF iterations.
            tol (float): Convergence tolerance.
            eta (float): Step size factor for HL-RF update.
        """
        super().__init__([g_node])
        self.g_node = g_node
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self._last_beta = None  # updated after each forward pass

    def forward(self, ctx: Context, *_):
        """
        Executes the HL-RF algorithm and returns Pf.

        Args:
            ctx (Context): Active computation context.

        Returns:
            torch.Tensor: Vector of failure probabilities (shape [B]).
        """
        u, v = ctx.u, ctx.v
        pf = FORMFunction.apply(self, self.g_node, u, v, self.max_iter, self.tol, self.eta)
        return pf

    def beta(self):
        """
        Returns the reliability indices β from the most recent forward pass.

        Returns:
            torch.Tensor: Tensor of β values (shape [B]) or None if not computed.
        """
        return self._last_beta
