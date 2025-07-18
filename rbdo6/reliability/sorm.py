# ============================================================
# File        : sorm.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Implements the Second-Order Reliability Method (SORM)
#               using Breitung's approximation. Supports full Hessian
#               or HVP-based eigenvalue estimation via Lanczos.
# ============================================================

import torch
import math
from torch.special import erf
from torch.linalg import norm
from ..core import Node, Context


class SORMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, form_node, g_node, g, u, v, max_iter, tol, eta, method, lanczos_k):
        with torch.enable_grad():
            B, _ = u.shape
            ctx.save_for_backward(u, v)

            pf = torch.zeros(B, dtype=u.dtype, device=u.device)
            u_star, beta = _run_hlrf(g_node, u, v, max_iter, tol, eta)

            # print("computed u*:")
            # print(u_star)

            kappa = _compute_curvatures(g_node, u_star, v, method, lanczos_k)

            form_node._beta = beta
            form_node._curvatures = kappa

            prod = torch.prod(1 + kappa / beta.unsqueeze(1), dim=1)
            pf = 0.5 * (1.0 - erf(beta / math.sqrt(2))) * prod.pow(-0.5)
            return pf

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        return (None, None, None, torch.zeros_like(u), torch.zeros_like(v),
                None, None, None, None, None)


class SORM(Node):
    def __init__(self, g_node, max_iter=50, tol=1e-6, eta=1.0, method="hvp", lanczos_k=10):
        u_node = Context.active().u_node
        v_node = Context.active().v_node
        super().__init__([g_node], side_inputs=[u_node, v_node])

        self.g_node = g_node
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.method = method
        self.lanczos_k = lanczos_k

        self._beta = None
        self._curvatures = None

    def forward(self, ctx: Context, g, u, v):
        pf = SORMFunction.apply(self, self.g_node, g, u.detach(), v.detach(),
                                self.max_iter, self.tol, self.eta,
                                self.method, self.lanczos_k)
        return pf

    def beta(self):
        return self._beta

    def curvatures(self):
        return self._curvatures


def _run_hlrf(g_node, u, v, max_iter, tol, eta):
    B, n = u.shape
    u_star = torch.zeros_like(u)
    beta = torch.zeros(B, device=u.device, dtype=u.dtype)

    for b in range(B):
        u_b = torch.zeros(n, device=u.device, dtype=u.dtype)
        v_b = v[b]

        g0_tmp = torch.zeros(1, n, device=u.device, dtype=u.dtype).requires_grad_(True)
        v_tmp = v_b.unsqueeze(0).detach().clone().requires_grad_(True)
        res0 = g_node.call(u=g0_tmp, v=v_tmp, grad=False)
        sign = -1.0 if res0["out"].item() < 0 else 1.0

        for _ in range(max_iter):
            u_tmp = u_b.unsqueeze(0).detach().clone().requires_grad_(True)
            v_tmp = v_b.unsqueeze(0).detach().clone().requires_grad_(True)
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

        u_star[b] = u_b
        beta[b] = sign * u_b.norm()

    return u_star, beta


def _compute_curvatures(g_node, u_star, v, method, lanczos_k):
    B, n = u_star.shape
    curvatures = []

    for i in range(B):
        ui = u_star[i].unsqueeze(0).requires_grad_(True)
        vi = v     [i].unsqueeze(0).requires_grad_(True)

        res = g_node.call(u=ui, v=vi, grad=True,
                          gradgrad_u=(method == "hessian"),
                          hvp_u=(None if method == "hessian" else torch.eye(n)))
        grad_u = res["grad_u"].squeeze(0)
        alpha = grad_u / norm(grad_u)
        T = _null_space(alpha.unsqueeze(0))

        if method == "hessian":
            H = res["hess_u"][0]
            H_tangent = T.T @ H @ T
        else:
            k_eff = min(lanczos_k, T.shape[1])  # do not exceed H_tangent size
            H_tangent = _lanczos_hvp(lambda v: g_node.call(u=ui, v=vi, hvp_u=v.unsqueeze(0))["hvp_u"].squeeze(0), T, k=k_eff)

        # print(H_tangent)

        eigvals = torch.linalg.eigvalsh(H_tangent).real
        curvatures.append(eigvals)

    return torch.stack(curvatures)


def _null_space(a):
    from scipy.linalg import null_space
    return torch.tensor(null_space(a.detach().cpu().numpy()), dtype=a.dtype, device=a.device)


def _lanczos_hvp(mv, T, k):
    n = T.shape[1]
    V = torch.zeros((n, k), device=T.device)
    alpha = torch.zeros(k, device=T.device)
    beta = torch.zeros(k - 1, device=T.device)

    v = torch.randn(n, device=T.device)
    v = v / norm(v)
    V[:, 0] = v

    w = T.T @ mv(T @ v)
    alpha[0] = torch.dot(w, v)
    w = w - alpha[0] * v

    for j in range(1, k):
        beta[j - 1] = norm(w)
        if beta[j - 1] == 0:
            break
        v = w / beta[j - 1]
        V[:, j] = v

        w = T.T @ mv(T @ v)
        w = w - beta[j - 1] * V[:, j - 1]
        alpha[j] = torch.dot(w, v)
        w = w - alpha[j] * v

    T_k = torch.diag(alpha) + torch.diag(beta, 1) + torch.diag(beta, -1)
    return T_k