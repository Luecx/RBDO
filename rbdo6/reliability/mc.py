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
        super().__init__([g_node])
        self.n_samples = n_samples
        self._samples = None
        self._pf = None

    def forward(self, ctx: Context, *_):
        """
        Performs Monte Carlo sampling in standard normal space and estimates Pf.

        Args:
            ctx (Context): Active computation context with .u, .v, and .random.

        Returns:
            torch.Tensor: Scalar tensor representing estimated failure probability.
        """
        device = ctx.v.device
        dtype = ctx.v.dtype

        # Generate samples in standard normal space for all random variables
        u_samples = torch.randn(self.n_samples, len(ctx.random), device=device, dtype=torch.float32)

        # Duplicate the design variables for batch evaluation
        v_batch = ctx.v.expand(self.n_samples, -1)

        # Evaluate limit state function g(u,v) for all samples
        res = self.inputs[0].call(u=u_samples, v=v_batch, grad=False)
        g_vals = res["out"].squeeze()

        # Store raw results and failure probability
        self._samples = g_vals.detach().cpu()
        self._pf = (g_vals < 0).float().mean().item()

        # Return estimated Pf as tensor
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

        # z-value for given confidence level (e.g., 1.96 for 95%)
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)

        # Standard deviation of Bernoulli estimator
        std = (self._pf * (1 - self._pf) / self.n_samples) ** 0.5
        delta = z * std

        return (self._pf - delta, self._pf + delta)

    @property
    def pf(self) -> float:
        """
        Returns the last computed failure probability.

        Returns:
            float: Pf from last forward pass.
        """
        return self._pf

    @property
    def samples(self) -> torch.Tensor:
        """
        Returns the raw evaluated g values from the last forward pass.

        Returns:
            torch.Tensor: Sampled g(u,v) values.
        """
        return self._samples
