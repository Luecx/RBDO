
# RBDO6 – A Lightweight Framework for Reliability-Based Design Optimization

**RBDO6** is a modular Python framework for conducting reliability-based design optimization (RBDO). It supports automatic differentiation and multiple failure probability estimators, and is built with PyTorch. The framework focuses on clarity and extensibility rather than performance or completeness.

### ✦ Features

* General node-based computation graph for design and random variables.
* Support for various random variable types (Normal, Uniform, Beta, Weibull, etc.).
* Correlation modeling via Cholesky-based Nataf transformation.
* Built-in estimators for failure probability:

  * **FORM** (First Order Reliability Method, HL-RF algorithm with autograd support)
  * **Monte Carlo Sampling** (MC)
  * **Importance Sampling** (IS)
* Optional support for black-box limit state functions with numerical differentiation.
* Hessian-vector products (HVP) and second-order derivatives (when available).
* Batch evaluation support (basic but functional).

### Limitations

* Some components (e.g. numerical gradients, black-box functions) are not optimized for large-scale usage.
* Error handling and user feedback are minimal.
* Only standard normal space is supported for uncertainty propagation (no Rosenblatt, etc.).
* Correlation modeling is limited to the Nataf transformation using constant covariance.
* Currently not intended for production-scale RBDO or high-performance applications.

---

## Example: Area Optimization with Reliability Constraint

A simple optimization problem minimizing the mean cross-section `mu_A` such that the failure probability is below a threshold using FORM:

```python
from rbdo6 import *
from scipy.optimize import minimize
import torch

with Context() as ctx:
    F = Normal(100, 1)
    A_mean = DesignVariable("mu_A", 10.0)
    A = Normal(A_mean, 0.1)
    sigma_max = Uniform(12, 13)

    ctx.corr = CorrelationMatrix()
    ctx.corr.compile()

    Z = NatafTransformation()
    X = Realization(Z, ctx.corr)

    class LimitState(Node):
        def forward(self, ctx, F, A, sigma_max):
            return sigma_max - F / A

    g = LimitState([X[F], X[A], X[sigma_max]])
    pf = FORM(g)

    def constraint(x):
        u = torch.zeros(len(ctx.random))
        v = torch.tensor(x, requires_grad=True)
        out = pf.call(u=u, v=v, grad=True)
        return 0.01 - out["out"].item(), -out["grad_v"].numpy()

    result = minimize(fun=lambda x: x[0],
                      x0=[10.0],
                      jac=lambda x: [1.0],
                      constraints=[{
                          "type": "ineq",
                          "fun": lambda x: constraint(x)[0],
                          "jac": lambda x: constraint(x)[1]
                      }],
                      bounds=[(5, 20)],
                      method="SLSQP")
```

---

## 📦 Installation

No pip package yet. Clone manually:

```bash
git clone https://github.com/Luecx/rbdo.git
cd rbdo
```

Install requirements:

```bash
pip install torch numpy scipy
```

---

## 📁 Structure Overview

```txt
rbdo6/
│
├── core/             # Context, node logic, computation engine
├── variable/         # Design and random variables
├── reliability/      # Estimators: FORM, MC, IS
├── correlation/      # Nataf transformation and correlation model
├── blackbox/         # Black-box support via numerical gradients
├── provider/         # Finite-difference Hessian-vector product
```

---

## ✅ Status

* Tested on small-scale academic problems
* Supports gradient-based optimization with reliability constraints
* Reasonably documented internally, but not fully production-ready

---

## 📄 License

MIT License. See `LICENSE` file.

---

Let me know if you want to include visuals, plots, or results from the example problem.
