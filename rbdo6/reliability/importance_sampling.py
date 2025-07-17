import math
import torch
from ..core import Node, Context
from scipy.stats import norm


class ImportanceSampling(Node):
    def __init__(self, g_node: Node, n_samples: int = 10000, max_iter: int = 50, tol: float = 1e-6, beta_thresh: float = 1.5):
        u_node = Context.active().u_node
        v_node = Context.active().v_node
        super().__init__([g_node, u_node, v_node])

        self.n_samples = n_samples
        self.max_iter = max_iter
        self.tol = tol
        self.beta_thresh = beta_thresh

        self._u_star = None
        self._pf = None
        self._samples = None
        self._weighted_samples = None

    def forward(self, ctx: Context, g: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        device = v.device
        dtype = v.dtype
        n_dim = len(ctx.random)

        _v = v.detach()
        _v_single = _v.squeeze(0)
        g_node = self.inputs[0]

        # Step 1: Compute u*
        u_star = self._compute_u_star(g_node, _v_single, n_dim, device)
        self._u_star = u_star

        # Step 2: Decide between IS and MC
        use_mc = u_star.norm().item() < self.beta_thresh
        if use_mc:
            print(f"[IS] Falling back to Monte Carlo (||u*|| = {u_star.norm().item():.3f})")
            u_samples = torch.randn(self.n_samples, n_dim, device=device)
        else:
            u_samples = self._sample_importance_distribution(u_star, device)

        v_batch = _v.expand(self.n_samples, -1)
        g_vals = g_node.call(u=u_samples, v=v_batch, grad=False)["out"].squeeze()
        indicator = (g_vals <= 0).float()
        self._samples = g_vals.detach().cpu()

        if use_mc:
            pf_est = indicator.mean()
            self._weighted_samples = indicator.detach().cpu()
            self._pf = pf_est.item()
            return pf_est.to(dtype)
        else:
            # Log-domain weights
            log_w = -torch.einsum("ij,j->i", u_samples, u_star) + 0.5 * u_star.dot(u_star)
            log_ind = torch.log(indicator + 1e-12)
            log_weighted = log_ind + log_w
            log_pf = torch.logsumexp(log_weighted, dim=0) - math.log(self.n_samples)

            # Store values
            self._weighted_samples = (log_weighted - log_pf).exp().detach().cpu()
            self._pf = log_pf.exp().item()

            # Debug
            print(f"log(w) mean/std/min/max: {log_w.mean().item():.3f} {log_w.std().item():.3f} "
                  f"{log_w.min().item():.3f} {log_w.max().item():.3f}")

            return torch.tensor(self._pf, device=device, dtype=dtype)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        if self._weighted_samples is None:
            raise RuntimeError("Importance Sampling not yet run.")
        z = norm.ppf(0.5 + level / 2)
        std = self._weighted_samples.std(unbiased=True).item() / math.sqrt(self.n_samples)
        delta = z * std
        return self._pf - delta, self._pf + delta

    def _compute_u_star(self, g_node: Node, v_b: torch.Tensor, n_dim: int, device) -> torch.Tensor:
        u_b = torch.zeros(n_dim, device=device)
        for _ in range(self.max_iter):
            u_tmp = u_b.unsqueeze(0).detach().requires_grad_(True)
            v_tmp = v_b.unsqueeze(0).detach().requires_grad_(True)
            res = g_node.call(u=u_tmp, v=v_tmp, grad=True)
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
        return u_b.detach()

    def _sample_importance_distribution(self, u_star: torch.Tensor, device) -> torch.Tensor:
        return torch.randn(self.n_samples, len(u_star), device=device) + u_star

    def _importance_weights(self, u_samples: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
        dot = torch.einsum("ij,j->i", u_samples, u_star)
        u_star_sq = u_star.dot(u_star).item()
        return torch.exp(-dot + 0.5 * u_star_sq)

    @property
    def pf(self) -> float:
        return self._pf

    @property
    def samples(self) -> torch.Tensor:
        return self._samples

    @property
    def weighted_samples(self) -> torch.Tensor:
        return self._weighted_samples

    @property
    def u_star(self) -> torch.Tensor:
        return self._u_star
