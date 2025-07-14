
from ..core import Node

class NatafTransformation(Node):
    def __init__(self):
        super().__init__()

    def forward(self, ctx):
        L = ctx.corr.get_L()  # [n, n]
        u = ctx.u  # [B, n]

        return u @ L.T  # [B, n]