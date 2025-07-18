# ============================================================
# File        : evaluate_thickness_pf.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Optimizes mu_t to minimize thickness under
#               the constraint that Pf <= 0.01 using FORM.
# ============================================================

import torch
import numpy as np
from scipy.optimize import minimize
from rbdo6 import *
from rbdo6.provider.grad.gradient_forward import ForwardDifference


class ThicknessLimitState(Node):
    def __init__(self, *nodes):
        super().__init__(list(nodes))

    def forward(self, ctx, *t_vals):
        # Constants
        L = 100.0
        E = 70000.0
        F = 100.0
        N = len(t_vals)

        # Positions and segment lengths
        x = torch.linspace(0, L, N + 1)
        dx = x[1] - x[0]

        # Convert thicknesses to moments of inertia: I = pi/4 * t^4
        t = torch.stack(t_vals, dim=1)  # shape: [B, N]
        I = (0.25 * torch.pi * t ** 4)  # shape: [B, N]

        # Initialize w and w'
        B = t.shape[0]
        w = torch.zeros((B, N + 1), dtype=t.dtype, device=t.device)
        w_prime = torch.zeros_like(w)

        # Moment function: M(x) = (1 - x / L) * F * L
        for i in range(N):
            x0 = x[i]
            x1 = x[i + 1]
            s = x1 - x0

            M0 = (1 - x0 / L) * F * L
            M1 = (1 - x1 / L) * F * L
            a = M0
            b = (M1 - M0) / s

            # Integrate:
            dw_prime = (s * a + b * 0.5 * s ** 2) / (E * I[:, i])
            dw = (0.5 * s ** 2 * a + b * (s ** 3 / 6)) / (E * I[:, i])

            w_prime[:, i + 1] = w_prime[:, i] + dw_prime
            w[:, i + 1] = w[:, i] + w_prime[:, i] * s + dw

        w_L = w[:, -1]  # displacement at x=L
        return 0.06 - w_L  # g < 0 is failure


def evaluate_thickness_pf():
    with Context() as ctx:
        # --- Design variable for mean thickness
        mu_t = DesignVariable("mu_t", 1.0)

        # --- Random variables: 5 local thicknesses (same std)
        sigma_t = 2.0
        t_rvs = [Normal(mu=mu_t, sigma=sigma_t) for _ in range(5)]

        # --- Correlation matrix
        corr = CorrelationMatrix()
        corr.compile()

        # --- Transformation and realization
        Z = NatafTransformation(corr)
        X = Realization(Z)

        g_node = ThicknessLimitState(*(X[t] for t in t_rvs))

        # --- Failure estimators
        pf_form = FORM(g_node)
        pf_mc   = MonteCarlo(g_node, n_samples=100000)

        PF_max = 0.01

        # --- Objective function: minimize mu_t
        def objective(x):
            return x[0], np.array([1.0])

        # --- FORM-based constraint: Pf(x) <= PF_max
        def constraint(x):
            u = torch.zeros((1, len(ctx.random)), requires_grad=True)
            v = torch.tensor([[x[0]]], dtype=torch.float32, requires_grad=True)
            out = pf_form.call(u=u, v=v, grad=True)
            value = PF_max - out["out"].item()
            grad = -out["grad_v"].detach().numpy()
            return value, grad

        def constraint2(x):
            u = torch.zeros((1, len(ctx.random)), requires_grad=True)
            v = torch.tensor([[x[0]]], dtype=torch.float32, requires_grad=True)
            pf_mc.call(u=u, v=v, grad=False)
            return PF_max - pf_mc.pf

        # --- MC-based constraint check (not used in opt)
        def constraint_mc(x):
            u = torch.zeros((1, len(ctx.random)), requires_grad=True)
            v = torch.tensor([[x[0]]], dtype=torch.float32, requires_grad=True)
            pf_mc.call(u=u, v=v, grad=False)
            return pf_mc.pf, *pf_mc.confidence_interval()

        for x in np.linspace(10, 14, 41):
            print(x, constraint([x]), constraint2([x]))

        # --- Run optimizer
        x0 = np.array([13.0])
        result = minimize(
            fun=lambda x: objective(x)[0],
            x0=x0,
            method='SLSQP',
            jac=lambda x: objective(x)[1],
            constraints=[{
                'type': 'ineq',
                'fun': lambda x: constraint2(x),
                # 'jac': lambda x: constraint(x)[1]
            }],
            bounds=[(5, 15.0)],
            options={'disp': True, 'ftol': 1e-1, 'maxiter': 100}
        )

        # --- Final evaluation
        mu_t_opt = result.x[0]
        pf_form_val = PF_max - constraint(result.x)[0]
        pf_mc_val, pf_mc_low, pf_mc_high = constraint_mc(result.x)

        print("\n✅ Optimization finished:")
        print(f"mu_t optimized  = {mu_t_opt:.4f}")
        print(f"Pf (FORM)       = {pf_form_val:.6f}")
        print(f"Pf (MC, 95% CI) = {pf_mc_val:.6f} ± {(pf_mc_high - pf_mc_low) * 0.5:.6f}")
        print(f"Success         = {result.success}")
        print(f"Status          = {result.message}")


if __name__ == "__main__":
    evaluate_thickness_pf()
