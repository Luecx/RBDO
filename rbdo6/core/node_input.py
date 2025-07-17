from .node import *
from .node_type import NodeKind  # If not already imported

class UNode(Node):
    def __init__(self):
        super().__init__(kind=NodeKind.U)

    def forward(self, ctx, u):
        return u


class VNode(Node):
    def __init__(self):
        super().__init__(kind=NodeKind.V)

    def forward(self, ctx, v):
        return v
