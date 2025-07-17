import torch
from ..core import Node, Context


class MonteCarlo(Node):
    """
    Monte Carlo Node for estimating the failure probability Pf = P[g(u,v) < 0].

    Attributes:
        n_samples (int): Number of Monte Carlo samples.

    Properties:
        samples (torch.Tensor): Evaluated g values from last call.
        pf (float): Estimated failure probability from last call.
    """

    def __init__(self, g_node: Node, n_samples: int = 10_000):
        """
        Initializes the Monte Carlo estimator.

        Args:
            g_node (Node): Limit state function node.
            n_samples (int): Number of Monte Carlo samples.
        """
        u_node = Context.active().u_node
        v_node = Context.active().v_node
        super().__init__([g_node, u_node, v_node])

        self.n_samples = n_samples
        self._samples = None
        self._pf = None

    def forward(self, ctx: Context, g_node, u: torch.Tensor, v: torch.Tensor):
        """
        Performs Monte Carlo sampling in standard normal space and estimates Pf.

        Args:
            ctx (Context): Active computation context with .random defined.
            v (torch.Tensor): Design variable tensor [1, n_v] or [B, n_v] — detached.

        Returns:
            torch.Tensor: Scalar tensor representing estimated failure probability.
        """

        _u = u.detach()
        _v = v.detach()

        device = v.device
        dtype  = torch.float32

        # Generate samples in standard normal space for all random variables
        n_rv = len(ctx.random)
        u_samples = torch.randn(self.n_samples, n_rv, device=device, dtype=dtype)

        # Evaluate g(u, v)
        g_node = self.inputs[0]
        res = g_node.call(u=u_samples, v=_v, grad=False)

        g_vals = res["out"].squeeze()

        # Store raw results and failure probability
        self._samples = g_vals.detach().cpu()
        self._pf = (g_vals < 0).float().mean().item()

        return torch.tensor(self._pf, device=device, dtype=dtype)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """
        Computes a symmetric confidence interval for the failure probability.

        Args:
            level (float): Confidence level (default: 0.95)

        Returns:
            Tuple (lower_bound, upper_bound): Error bars around Pf
        """
        if self._pf is None:
            raise RuntimeError("Monte Carlo not yet evaluated.")

        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        std = (self._pf * (1 - self._pf) / self.n_samples) ** 0.5
        delta = z * std

        return (self._pf - delta, self._pf + delta)

    @property
    def pf(self) -> float:
        """Returns the last computed failure probability."""
        return self._pf

    @property
    def samples(self) -> torch.Tensor:
        """Returns the raw evaluated g values from the last forward pass."""
        return self._samples
