# ============================================================
# File        : hvp_fdiff.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Forward-difference HVP approximation.
# ============================================================

import torch
from .hvp_provider import HVPProvider


class HVP_FDiff_Provider(HVPProvider):
    def __init__(self, eps=1e-2):
        self.eps = eps

    def __call__(self, node, vec, u, v, wrt="u"):
        """
        Approximates H @ vec using forward finite differences of gradients.

        Args:
            node (Node): The node whose output y = f(u, v) is scalar per batch.
            vec (Tensor): [B, D] vector to multiply the Hessian with.
            u (Tensor): [B, D_u] standard normal inputs.
            v (Tensor): [B, D_v] design variable inputs.
            wrt (str): "u" or "v" – the variable to differentiate w.r.t.

        Returns:
            Tensor: [B, D] H @ vec approximation.
        """
        B = vec.shape[0]
        result = []

        for b in range(B):
            u0 = u[b].detach().clone()
            v0 = v[b].detach().clone()
            v_ = vec[b]

            def eval_grad(u_eval, v_eval):
                out = node.call(
                    u=u_eval.unsqueeze(0).requires_grad_(),
                    v=v_eval.unsqueeze(0).requires_grad_(),
                    grad=True
                )
                if wrt == "u":
                    return out["grad_u"].squeeze(0)
                elif wrt == "v":
                    return out["grad_v"].squeeze(0)
                else:
                    raise ValueError(f"Invalid wrt value: {wrt}")

            if wrt == "u":
                g0 = eval_grad(u0, v0)
                g1 = eval_grad(u0 + self.eps * v_, v0)
            elif wrt == "v":
                g0 = eval_grad(u0, v0)
                g1 = eval_grad(u0, v0 + self.eps * v_)
            else:
                raise ValueError(f"Invalid wrt value: {wrt}")

            result.append((g1 - g0) / self.eps)

        return torch.stack(result)
