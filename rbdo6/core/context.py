import torch


# ===============================================================
# Context
# ===============================================================

class Context:
    _active = None

    def __init__(self):
        self.design = []
        self.random = []
        self.corr = None
        self.u = None
        self.v = None
        self.stats = {
            "forward_calls": 0,
            "blackbox_forward": 0,
            "blackbox_backward": 0
        }

    def __enter__(self):
        Context._active = self
        return self

    def __exit__(self, *a):
        Context._active = None

    def register_design(self, var):
        var._id = len(self.design)
        self.design.append(var)

    def register_random(self, var):
        var._id = len(self.random)
        self.random.append(var)

    def set_inputs(self, u, v):
        if u.ndim == 1:
            u = u.unsqueeze(0)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        if not u.requires_grad:
            u = u.clone().detach().requires_grad_(True)
        if not v.requires_grad:
            v = v.clone().detach().requires_grad_(True)

        self.u = u
        self.v = v

    @staticmethod
    def active():
        if Context._active is None:
            raise RuntimeError("No active context.")
        return Context._active