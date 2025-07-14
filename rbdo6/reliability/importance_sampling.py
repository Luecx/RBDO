# ============================================================
# File        : importance_sampling.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Importance Sampling with automatic u* estimation using HLRF.
# ============================================================

import math
import torch
from ..core import Node, Context


class ImportanceSampling(Node):
    """
    Importance Sampling node using u* computed from HL-RF (same as FORM).
    Evaluates Pf = E_q[𝟙[g(u,v) < 0] w(u)], where q = N(u*, I) and
    w(u) = φ(u) / q(u) = exp(-u*·(u - 0.5·u*)).

    Attributes:
        g_node (Node): Limit state function node.
        n_samples (int): Number of importance samples.
        max_iter (int): Max iterations for HL-RF.
        tol (float): Convergence tolerance for HL-RF.
    """

    def __init__(self, g_node: Node, n_samples: int = 10000, max_iter: int = 50, tol: float = 1e-6):
        super().__init__([g_node])
        self.g_node = g_node
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.tol = tol
        self._u_star = None
        self._pf = None
        self._samples = None

    def forward(self, ctx: Context, *_):
        """
        Runs HL-RF to find u*, then estimates Pf via importance sampling.

        Returns:
            torch.Tensor: Estimated failure probability.
        """
        device = ctx.v.device
        n_dim  = len(ctx.random)

        # === Compute u* via HL-RF
        u_b = torch.zeros(n_dim, device=device, dtype=torch.float32)
        v_b = ctx.v.squeeze(0).detach()

        for _ in range(self.max_iter):
            u_tmp = u_b.unsqueeze(0).clone().detach().requires_grad_(True)
            v_tmp = v_b.unsqueeze(0).clone().detach().requires_grad_(True)

            Context.active().set_inputs(u_tmp, v_tmp)
            res = self.g_node.call(u=u_tmp, v=v_tmp, grad=True)
            g_val = res["out"].item()
            du = res["grad_u"].squeeze(0)
            norm2 = du.dot(du).item()

            if abs(g_val) < self.tol or norm2 == 0.0:
                break

            lam = (du @ u_b - g_val) / norm2
            u_new = u_b + (lam * du - u_b)
            if (u_new - u_b).norm().item() < self.tol:
                u_b = u_new
                break
            u_b = u_new

        self._u_star = u_b.detach()

        # === Importance sampling
        u_samples = torch.randn(self.n_samples, n_dim, device=device, dtype=torch.float32) + self._u_star
        v_batch   = ctx.v.expand(self.n_samples, -1)

        res = self.g_node.call(u=u_samples, v=v_batch, grad=False)
        g_vals = res["out"].squeeze()

        # Importance weight: w(u) = exp(-u*·(u - 0.5 u*))
        dot = torch.einsum("ij,j->i", u_samples, self._u_star)
        u_star_sq = torch.dot(self._u_star, self._u_star).item()
        weights = torch.exp(-dot + 0.5 * u_star_sq)

        indicator = (g_vals < 0).float()
        weighted = indicator * weights
        pf = weighted.mean()

        self._samples = g_vals.detach().cpu()
        self._pf = pf.item()
        return torch.tensor(self._pf, device=device, dtype=torch.float32)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        Computes a symmetric confidence interval for the IS estimate using the
        sample standard deviation of the weighted estimator.
        """
        if self._samples is None or self._u_star is None:
            raise RuntimeError("Importance Sampling not yet run.")

        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)

        u = self._u_star
        u_samples = self._samples.to(dtype=u.dtype, device=u.device)
        n = self.n_samples

        # Reconstruct weights and indicators
        u_raw = torch.randn(n, len(u), device=u.device) + u
        dot = torch.einsum("ij,j->i", u_raw, u)
        u_norm2 = torch.dot(u, u).item()
        weights = torch.exp(-dot + 0.5 * u_norm2)

        # Re-evaluate g (reusing stored values if you stored them)
        v_batch = Context.active().v.expand(n, -1)
        res = self.inputs[0].call(u=u_raw, v=v_batch, grad=False)
        g_vals = res["out"].squeeze()
        indicator = (g_vals < 0).float()

        y = indicator * weights
        std = y.std(unbiased=True).item() / math.sqrt(n)
        delta = z * std

        return self._pf - delta, self._pf + delta

    @property
    def pf(self) -> float:
        """Returns last computed Pf."""
        return self._pf

    @property
    def samples(self) -> torch.Tensor:
        """Returns last evaluated g values."""
        return self._samples

    @property
    def u_star(self) -> torch.Tensor:
        """Returns computed u* from HL-RF iteration."""
        return self._u_star
