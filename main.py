# ============================================================
# File        : optimize_mu_A.py
# Project     : Reliability-Based Design Optimization
# Author      : Finn Eggers
# Description : Optimizes mu_A to minimize area under the constraint
#               that the failure probability Pf <= 0.01 using FORM.
#               sigma_max is modeled as a normal distribution.
#               The limit state uses numerical differentiation.
# ============================================================

import torch
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from rbdo6 import *
from rbdo6.provider.grad.gradient_forward import ForwardDifference

def optimize_mu_A():
    with Context() as ctx:
        # --- Fixed parameters
        F_mean = 0.9
        F_std = 0.1
        A_std = 0.1
        PF_max = 0.1

        # --- Design variable
        A_mean = DesignVariable("mu_A", 0.6)

        # --- Random variables
        F = Normal(mu=F_mean, sigma=F_std)
        A = Normal(mu=A_mean, sigma=A_std)
        sigma_max = Normal(0.4, 0.1)

        # --- Correlation matrix
        corr = CorrelationMatrix()
        corr.compile()

        # --- Transformation and realization
        Z = NatafTransformation(corr)
        X = Realization(Z)

        class CurvedLimitState(Node):
            def __init__(self, x1_node, x2_node, x3_node):
                super().__init__([x1_node, x2_node, x3_node])

            def forward(self, ctx, x1, x2, x3):
                return 1.0 - x1 ** 2 - 0.5 * x2 ** 2 - 0.25 * x3 ** 2

        # --- Define custom Node for limit state: g = sigma_max - F / A
        class StructuralLimitState(Node):
            def __init__(self, force_node, area_node, sigmax_node):
                super().__init__([force_node, area_node, sigmax_node])

            def forward(self, ctx, force, area, sig_max):
                # Break autograd, use numerical differentiation
                force   = force
                area    = area
                sig_max = sig_max
                return sig_max - force / area

        g_node = CurvedLimitState(X[F], X[A], X[sigma_max])

        # --- Failure probability estimators
        pf_form = FORM(g_node)
        pf_sorm = SORM(g_node, method="hvp")
        pf_mc   = MonteCarlo(g_node)


        u = torch.zeros((1,3), requires_grad=True)
        v = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)

        print(pf_form.call(u, v, grad=False))
        print(pf_sorm.call(u, v, grad=False))
        print(pf_mc.call(u, v, grad=False))

        # pf_form.plot()


        # pf_mc = MonteCarlo(g_node, n_samples=10000)
        # pf_is = ImportanceSampling(g_node, n_samples=10000)
        #
        # # --- Objective function: minimize mu_A
        # def objective(x):
        #     return x[0], np.array([1.0])
        #
        # # --- FORM-based constraint: Pf(x) <= PF_max
        # def constraint(x):
        #     u = torch.zeros((1, len(ctx.random)), requires_grad=True)
        #     v = torch.tensor([x], dtype=torch.float32, requires_grad=True)
        #     out = pf_form.call(u=u, v=v, grad=True)
        #     value = PF_max - out["out"].item()
        #     grad = -out["grad_v"].detach().numpy()
        #     return value, grad
        #
        # # --- Monte Carlo evaluation at final point
        # def constraint_mc(x):
        #     u = torch.zeros((1, len(ctx.random)), requires_grad=True)
        #     v = torch.tensor([x], dtype=torch.float32, requires_grad=True)
        #     pf_mc.call(u=u, v=v, grad=False)
        #     return pf_mc.pf, *pf_mc.confidence_interval()
        #
        # def constraint_is(x):
        #     u = torch.zeros((1, len(ctx.random)), requires_grad=True)
        #     v = torch.tensor([x], dtype=torch.float32, requires_grad=True)
        #     pf_is.call(u=u, v=v, grad=False)
        #     return pf_is.pf, *pf_is.confidence_interval()
        #
        # # --- Run optimizer
        # x0 = np.array([10.0])
        # result = minimize(
        #     fun=lambda x: objective(x)[0],
        #     x0=x0,
        #     method='SLSQP',
        #     jac=lambda x: objective(x)[1],
        #     constraints=[{
        #         'type': 'ineq',
        #         'fun': lambda x: constraint(x)[0],
        #         'jac': lambda x: constraint(x)[1]
        #     }],
        #     bounds=[(8.0, 20.0)],
        #     options={'disp': True, 'ftol': 1e-6, 'maxiter': 100}
        # )
        #
        # # --- Final evaluation
        # mu_A_opt = result.x[0]
        # pf_form_val = PF_max - constraint(result.x)[0]
        # pf_mc_val, pf_mc_low, pf_mc_high = constraint_mc(result.x)
        # pf_is_val, pf_is_low, pf_is_high = constraint_is(result.x)
        #
        # print("\n✅ Optimization finished:")
        # print(f"mu_A optimized  = {mu_A_opt:.4f}")
        # print(f"Pf (FORM)       = {pf_form_val:.6f}")
        # print(f"Pf (MC, 95% CI) = {pf_mc_val:.6f}  ±  {(pf_mc_high - pf_mc_low) * 0.5:.6f}")
        # print(f"Pf (IS, 95% CI) = {pf_is_val:.6f}  ±  {(pf_is_high - pf_is_low) * 0.5:.6f}")
        # print(f"Success         = {result.success}")
        # print(f"Status          = {result.message}")

if __name__ == "__main__":
    optimize_mu_A()
