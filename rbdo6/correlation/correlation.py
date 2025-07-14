
import torch
from ..core import Context

class CorrelationMatrix:
    def __init__(self):
        self.random_vars = Context.active().random
        self.n = len(self.random_vars)
        self.R = torch.eye(self.n)
        self.L = None
        self._dirty = True

    def set_correlation(self, r1, r2, rho):
        i, j = r1._id, r2._id
        self.R[i, j] = self.R[j, i] = rho
        self._dirty = True

    def compile(self):
        self.L = torch.linalg.cholesky(self.R).to(dtype=torch.float32)
        self._dirty = False

    def get_L(self):
        if self._dirty:
            raise RuntimeError("CorrelationMatrix not compiled.")
        return self.L
