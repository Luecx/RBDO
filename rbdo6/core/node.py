# ============================================================
# File        : node.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Base class for all computational nodes in the
#               computation graph. Handles forward evaluation,
#               gradient/Hessian computation, and HVPs.
# ============================================================

import torch
import matplotlib.pyplot as plt

from torch.autograd.functional import hvp
from graphviz import Digraph
from io import BytesIO
from PIL import Image

from .context import Context
from .node_type import *
from ..provider.grad import GradientProvider, CustomGradFunction
from ..provider.hvp import HVPProvider


class Node:
    """
    Base class for computational graph nodes in the RBDO framework.

    Nodes can have dependencies (`inputs`), side inputs (non-differentiated),
    and support analytic or numeric differentiation including:
    - Gradients
    - Hessians
    - Hessian-vector products (HVP)

    Attributes:
        inputs (list): Primary input nodes or tensors.
        side_inputs (list): Non-differentiated auxiliary inputs.
        kind (NodeKind): Type of this node (STANDARD, U, V).
        grad_provider (GradientProvider): Optional custom gradient handler.
        hesse_provider (callable): Optional custom Hessian handler.
        hvp_provider (HVPProvider): Optional custom HVP handler.
        grad_mode (DerivativeMode): ANALYTIC or NUMERIC.
    """

    def __init__(self, inputs=None, side_inputs=None,
                 kind=NodeKind.STANDARD,
                 grad_provider=None,
                 hesse_provider=None,
                 hvp_provider=None):
        self.inputs = inputs if inputs else []
        self.side_inputs = side_inputs if side_inputs else []
        self.kind = kind

        self.grad_provider = grad_provider
        self.hesse_provider = hesse_provider
        self.hvp_provider = hvp_provider

        self.grad_mode = DerivativeMode.NUMERIC if grad_provider or any(
            isinstance(inp, Node) and inp.grad_mode == DerivativeMode.NUMERIC
            for inp in self.inputs
        ) else DerivativeMode.ANALYTIC

        self._cache = None
        self._cache_key = None

    def forward(self, ctx, *args) -> torch.Tensor:
        """
        Override this method to implement the node's forward logic.

        Args:
            ctx (Context): Active computation context.
            *args: Combined inputs and side inputs.

        Returns:
            torch.Tensor: Batched output tensor [B, ...]
        """
        raise NotImplementedError

    def get_inputs(self, u, v, _call_id):
        """
        Evaluates and collects all tracked and side inputs.

        Args:
            u (torch.Tensor): Standard normal input [B, n_u]
            v (torch.Tensor): Design variable input [B, n_v]
            _call_id (int): Unique call ID for cache handling

        Returns:
            tuple: (tracked_inputs, side_inputs)
        """
        tracked = []
        untracked = []

        for inp in self.inputs:
            if isinstance(inp, Node):
                if inp.kind == NodeKind.U:
                    tracked.append(u)
                elif inp.kind == NodeKind.V:
                    tracked.append(v)
                else:
                    tracked.append(inp.call(u=u, v=v, _call_id=_call_id)["out"])
            else:
                tracked.append(inp)

        for inp in self.side_inputs:
            if isinstance(inp, Node):
                val = (
                    u.detach() if inp.kind == NodeKind.U else
                    v.detach() if inp.kind == NodeKind.V else
                    inp.call(u=u, v=v, _call_id=_call_id)["out"].detach()
                )
                untracked.append(val)
            elif isinstance(inp, torch.Tensor):
                untracked.append(inp.detach())
            else:
                untracked.append(inp)

        return tracked, untracked

    def call(self, u=None, v=None,
             grad=False, gradgrad_u=False, gradgrad_v=False,
             hvp_u=None, hvp_v=None, _call_id=None) -> dict:
        """
        Evaluates the node, optionally computing gradients, Hessians, or HVPs.

        Args:
            u (torch.Tensor): Input from standard normal space [B, n_u]
            v (torch.Tensor): Design variable input [B, n_v]
            grad (bool): If True, compute gradients wrt u and v
            gradgrad_u (bool): If True, compute Hessian wrt u
            gradgrad_v (bool): If True, compute Hessian wrt v
            hvp_u (torch.Tensor): If provided, compute HVP wrt u
            hvp_v (torch.Tensor): If provided, compute HVP wrt v
            _call_id (int): Optional ID for avoiding duplicate calls

        Returns:
            dict: Dictionary containing outputs and optionally derivatives
        """
        root = _call_id is None
        if root:
            self._ensure_batched(u, v)

        ctx = Context.active()
        result = {}

        if _call_id is None:
            _call_id = ctx._next_call_id()

        if _call_id != self._cache_key:
            tracked_inputs, side_inputs = self.get_inputs(u, v, _call_id)
            inputs = tracked_inputs + side_inputs

            ctx.stats["forward_calls"] += 1

            if self.grad_provider is not None:
                out = CustomGradFunction.apply(
                    self.grad_provider,
                    lambda *x: self.forward(ctx, *x, *side_inputs),
                    *tracked_inputs
                )
            else:
                out = self.forward(ctx, *inputs)

            self._cache_key = _call_id
            self._cache = out
        else:
            out = self._cache

        result["out"] = out

        if not root:
            return result

        # --- Gradients ---
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
                result["grad_u"] = u.grad.clone() if u.grad is not None else None
                result["grad_v"] = v.grad.clone() if v.grad is not None else None
            else:
                raise RuntimeError("Cannot compute gradients due to disconnection")

        # --- Second-order ---
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
            raise ValueError(
                "Input `u` must be 2D (batched). Got shape {}. "
                "Use u.unsqueeze(0) to add a batch dimension.".format(tuple(u.shape))
            )
        if v is not None and v.ndim == 1:
            raise ValueError(
                "Input `v` must be 2D (batched). Got shape {}. "
                "Use v.unsqueeze(0) to add a batch dimension.".format(tuple(v.shape))
            )

    def _compute_hessian_u(self, u, v):
        if self.hesse_provider:
            return self.hesse_provider(self, u, v, wrt="u")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute Hessian for numerically differentiated node without a hesse_provider.")
        return torch.stack([
            torch.autograd.functional.hessian(
                lambda uu: self.call(u=uu.unsqueeze(0), v=v[i].unsqueeze(0))["out"].squeeze(0),
                u[i].detach().requires_grad_(), create_graph=True
            ) for i in range(u.shape[0])
        ])

    def _compute_hessian_v(self, u, v):
        if self.hesse_provider:
            return self.hesse_provider(self, u, v, wrt="v")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute Hessian for numerically differentiated node without a hesse_provider.")
        return torch.stack([
            torch.autograd.functional.hessian(
                lambda vv: self.call(u=u[i].unsqueeze(0), v=vv.unsqueeze(0))["out"].squeeze(0),
                v[i].detach().requires_grad_(), create_graph=True
            ) for i in range(v.shape[0])
        ])

    def _compute_hvp_u(self, u, v, hvp_vecs):
        if self.hvp_provider:
            return self.hvp_provider(self, hvp_vecs, u, v, wrt="u")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute HVP for numerically differentiated node without a hvp_provider.")
        return torch.stack([
            hvp(
                lambda uu: self.call(u=uu.unsqueeze(0), v=v[i].unsqueeze(0))["out"].squeeze(0),
                u[i].detach().requires_grad_(), hvp_vecs[i]
            )[1] for i in range(u.shape[0])
        ])

    def _compute_hvp_v(self, u, v, hvp_vecs):
        if self.hvp_provider:
            return self.hvp_provider(self, hvp_vecs, u, v, wrt="v")
        if self.grad_mode == DerivativeMode.NUMERIC:
            raise RuntimeError("Cannot compute HVP for numerically differentiated node without a hvp_provider.")
        return torch.stack([
            hvp(
                lambda vv: self.call(u=u[i].unsqueeze(0), v=vv.unsqueeze(0))["out"].squeeze(0),
                v[i].detach().requires_grad_(), hvp_vecs[i]
            )[1] for i in range(v.shape[0])
        ])

    def set_grad_provider(self, provider: GradientProvider):
        """
        Sets a custom gradient provider and switches to numeric mode.

        Args:
            provider (GradientProvider): Gradient override.
        """
        if not isinstance(provider, GradientProvider):
            raise TypeError("grad_provider must be an instance of GradientProvider.")
        self.grad_provider = provider
        self.grad_mode = DerivativeMode.NUMERIC

    def set_hvp_provider(self, provider: HVPProvider):
        """
        Sets a custom Hessian-vector-product (HVP) provider.

        Args:
            provider (HVPProvider): Custom HVP logic.
        """
        if not isinstance(provider, HVPProvider):
            raise TypeError("hvp_provider must be an instance of HVPProvider.")
        self.hvp_provider = provider

    def set_hesse_provider(self, func):
        """
        Sets a custom Hessian provider (callable).

        Args:
            func (callable): Hessian computation function.
        """
        self.hesse_provider = func

    def plot(self, figsize=(10, 6), dpi=100):
        """
        Visualizes the computation graph using Graphviz and renders it with matplotlib.

        Args:
            figsize (tuple): Figure size for matplotlib.
            dpi (int): Resolution for rendering the image.
        """
        dot = Digraph(comment="RBDO Computation Graph")
        dot.attr(rankdir="LR", fontsize="12")

        visited = set()
        counter = {"id": 0}
        node_map = {}

        def node_id(node):
            if node not in node_map:
                node_map[node] = f"node_{counter['id']}"
                counter["id"] += 1
            return node_map[node]

        def add_node(node):
            if node in visited or not isinstance(node, Node):
                return
            visited.add(node)
            nid = node_id(node)

            label = node.__class__.__name__
            flags = []
            info = []
            if node.grad_provider:
                name = type(node.grad_provider).__name__
                info.append(f"grad = {name}")
            if node.hvp_provider:
                name = type(node.hvp_provider).__name__
                info.append(f"hvp = {name}")
            if node.hesse_provider:
                flags.append("H")
            if info:
                label += "\\n" + "\\n".join(info)

            dot.node(nid, label=label, shape="box")

            for inp in node.inputs:
                if isinstance(inp, Node):
                    add_node(inp)
                    dot.edge(node_id(inp), nid)

            for inp in node.side_inputs:
                if isinstance(inp, Node):
                    add_node(inp)
                    dot.edge(node_id(inp), nid, style="dotted")

        add_node(self)

        # Render to PNG in memory
        img_data = dot.pipe(format="png")
        image = Image.open(BytesIO(img_data))

        # Show using matplotlib
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()