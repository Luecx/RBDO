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
    """
    Custom autograd function for FORM, computing Pf ≈ Φ(-β) using HL-RF iteration.

    This function tracks gradients w.r.t. the limit state value g.
    """

    @staticmethod
    def forward(ctx, form_node, g_node, g, u, v, max_iter, tol, eta):
        """
        Forward pass of the FORM autograd function.

        Args:
            form_node (FORM): The calling FORM node.
            g_node (Node): The limit state function node g(u, v).
            g (torch.Tensor): Limit state values [B].
            u (torch.Tensor): Standard normal samples [B, n_u].
            v (torch.Tensor): Design variable samples [B, n_v].
            max_iter (int): Max iterations for HL-RF.
            tol (float): Convergence tolerance.
            eta (float): Step scaling factor.

        Returns:
            torch.Tensor: Failure probability estimates [B].
        """
        B, n_u = u.shape

        # Detach inputs for manual tracking
        u = u.detach()
        v = v.detach()

        # Store data for backward
        ctx.g_node = g_node
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.eta = eta
        ctx.form_node = form_node

        # Prepare storage for u*, β, and gradients
        ctx.u_star = torch.zeros_like(u)
        beta = torch.zeros(B, device=u.device, dtype=u.dtype)
        ctx.du = []
        ctx.dv = []

        pf_out = torch.zeros(B, device=u.device, dtype=u.dtype)

        # --- HL-RF Loop over batch
        with torch.enable_grad():
            for b in range(B):
                u_b = torch.zeros(n_u, device=u.device, dtype=u.dtype)
                v_b = v[b]

                # Determine failure sign using dummy g call
                g0_tmp = torch.zeros(1, n_u, device=u.device, dtype=u.dtype).requires_grad_(True)
                v_tmp = v_b.unsqueeze(0).detach().clone().requires_grad_(True)
                res0 = g_node.call(u=g0_tmp, v=v_tmp, grad=False)
                sign = -1.0 if res0["out"].item() < 0 else 1.0

                # --- HL-RF iteration
                for _ in range(max_iter):
                    u_tmp = u_b.unsqueeze(0).detach().clone().requires_grad_(True)
                    v_tmp = v_b.unsqueeze(0).detach().clone().requires_grad_(True)

                    res = g_node.call(u=u_tmp, v=v_tmp, grad=True)
                    g_val = res["out"].item()
                    du = res["grad_u"].squeeze(0)
                    norm2 = du.dot(du).item()

                    if abs(g_val) < tol or norm2 == 0.0:
                        break

                    # HL-RF update step
                    lam = (du @ u_b - g_val) / norm2
                    u_new = u_b + eta * (lam * du - u_b)

                    if (u_new - u_b).norm().item() < tol:
                        u_b = u_new
                        break

                    u_b = u_new

                # Final g evaluation at converged u*
                u_tmp = u_b.unsqueeze(0).detach().clone().requires_grad_(True)
                v_tmp = v_b.unsqueeze(0).detach().clone().requires_grad_(True)

                res = g_node.call(u=u_tmp, v=v_tmp, grad=True)
                du = res["grad_u"].squeeze(0)
                dv = res["grad_v"].squeeze(0)

                # Compute β and Pf
                β = sign * u_b.norm()
                pf_val = 0.5 * (1.0 - erf(β / math.sqrt(2.0)))

                pf_out[b] = pf_val
                ctx.u_star[b] = u_b
                beta[b] = β
                ctx.du.append(du)
                ctx.dv.append(dv)

        # Cache for FORM node access
        form_node._last_u_star = ctx.u_star.detach().clone()
        form_node._last_beta = beta.detach().clone()
        ctx.save_for_backward(beta)

        return pf_out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for FORM — computes dPf/dg using chain rule.

        Args:
            grad_out (torch.Tensor): Upstream gradient [B].

        Returns:
            Tuple: Gradients w.r.t. g; other entries are None.
        """
        (beta,) = ctx.saved_tensors
        B = grad_out.shape[0]
        grad_g = torch.zeros(B, device=grad_out.device)

        for b in range(B):
            norm_du = ctx.du[b].norm().item()
            if norm_du == 0.0:
                dpf_dg = 0.0
            else:
                dpf_dg = -math.exp(-beta[b].item()**2 / 2.0) / math.sqrt(2.0 * math.pi) / norm_du

            grad_g[b] = dpf_dg * grad_out[b]

        # Only gradient w.r.t. g is returned
        return None, None, grad_g, None, None, None, None, None


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
        u_node = Context.active().u_node
        v_node = Context.active().v_node
        super().__init__([g_node], side_inputs=[u_node, v_node])

        self.g_node = g_node
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta

        self._last_beta = None
        self._last_u_star = None

    def forward(self, ctx: Context, g, u, v):
        """
        Executes the forward FORM computation using autograd wrapper.

        Args:
            ctx (Context): The execution context.
            g (torch.Tensor): Limit state values [B].
            u (torch.Tensor): Standard normal inputs [B, n_u].
            v (torch.Tensor): Design inputs [B, n_v].

        Returns:
            torch.Tensor: Failure probability estimates [B].
        """
        pf = FORMFunction.apply(self, self.g_node, g, u.detach(), v.detach(),
                                self.max_iter, self.tol, self.eta)
        return pf

    def beta(self):
        """
        Returns the β indices from the last forward pass.

        Returns:
            torch.Tensor: Tensor [B] of reliability indices.
        """
        return self._last_beta

    def u_star(self):
        """
        Returns the u* design points from the last forward pass.

        Returns:
            torch.Tensor: Tensor of shape [B, n_u] containing HL-RF solution points.
        """
        return self._last_u_star
