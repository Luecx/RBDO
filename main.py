# ============================================================
# File        : optimize_mu_A.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Optimizes mu_A to minimize area under the constraint
#               that the failure probability Pf <= 0.01 using FORM.
#               sigma_max is modeled as a uniform distribution in [11, 13].
# ============================================================

import torch
import numpy as np
from scipy.optimize import minimize
from rbdo6 import *


def optimize_mu_A():
    with Context() as ctx:

        # --- Fixed parameters
        F_mean = 100.0
        F_std  = 1.0
        A_std  = 0.1
        PF_max = 0.01

        # --- Design variable
        A_mean = DesignVariable("mu_A", 10.0)

        # --- Random variables
        F         = Normal(mu=F_mean, sigma=F_std)
        A         = Normal(mu=A_mean, sigma=A_std)
        sigma_max = Uniform(12, 13)

        # --- Correlation matrix
        ctx.corr = CorrelationMatrix()
        ctx.corr.compile()

        # --- Transformation and realization
        Z = NatafTransformation()
        X = Realization(Z, ctx.corr)

        # --- Limit state function g = sigma_max - F / A
        class StructuralLimitState(Node):
            def __init__(self, force_node, area_node, sigma_max_node):
                super().__init__([force_node, area_node, sigma_max_node])

            def forward(self, ctx, force, area, sigma_max):
                return sigma_max - force / area

        g_node  = StructuralLimitState(X[F], X[A], X[sigma_max])
        pf_form = FORM(g_node)
        pf_mc   = MonteCarlo(g_node, n_samples=10000)
        pf_is   = ImportanceSampling(g_node, n_samples=100)


        # --- Objective function: minimize mu_A
        def objective(x):
            return x[0], np.array([1.0])

        # --- FORM-based constraint: Pf(x) <= PF_max
        def constraint(x):
            u = torch.zeros(len(ctx.random), requires_grad=True, dtype=torch.float32)
            v = torch.tensor(x, requires_grad=True)
            out = pf_form.call(u=u, v=v, grad=True)
            value = PF_max - out["out"].item()
            grad  = -out["grad_v"].detach().numpy()
            return value, grad

        # --- Monte Carlo evaluation at final point
        def constraint_mc(x):
            u = torch.zeros(len(ctx.random), requires_grad=True, dtype=torch.float32)
            v = torch.tensor(x, requires_grad=True)
            out = pf_mc.call(u=u, v=v, grad=False)
            pf_val = pf_mc.pf
            pf_low, pf_high = pf_mc.confidence_interval()
            return pf_val, pf_low, pf_high
        
        def constraint_is(x):
            u = torch.zeros(len(ctx.random), requires_grad=True, dtype=torch.float32)
            v = torch.tensor(x, requires_grad=True)
            out = pf_is.call(u=u, v=v, grad=False)
            pf_val = pf_is.pf
            pf_low, pf_high = pf_is.confidence_interval()
            return pf_val, pf_low, pf_high

        # --- Run optimizer
        x0 = np.array([10.0])
        result = minimize(
            fun=lambda x: objective(x)[0],
            x0=x0,
            method='SLSQP',
            jac=lambda x: objective(x)[1],
            constraints=[{
                'type': 'ineq',
                'fun': lambda x: constraint(x)[0],
                'jac': lambda x: constraint(x)[1]
            }],
            bounds=[(8.0, 20.0)],
            options={'disp': True, 'ftol': 1e-6, 'maxiter': 100}
        )

        # --- Final evaluation
        mu_A_opt = result.x[0]
        pf_form_val = PF_max - constraint(result.x)[0]
        pf_mc_val, pf_mc_low, pf_mc_high = constraint_mc(result.x)
        pf_is_val, pf_is_low, pf_is_high = constraint_is(result.x)

        # --- Print results
        print("\n✅ Optimization finished:")
        print(f"mu_A optimized  = {mu_A_opt:.4f}")
        print(f"Pf (FORM)       = {pf_form_val:.6f}")
        print(f"Pf (MC, 95% CI) = {pf_mc_val:.6f}  ±  {0.5 * (pf_mc_high - pf_mc_low):.6f}")
        print(f"Pf (IS, 95% CI) = {pf_is_val:.6f}  ±  {0.5 * (pf_is_high - pf_is_low):.6f}")
        print(f"Success         = {result.success}")
        print(f"Status          = {result.message}")


if __name__ == "__main__":
    optimize_mu_A()
