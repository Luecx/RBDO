from .node import *

class IndexNode(Node):
    def __init__(self, index, source_node):
        super().__init__([source_node])
        self.index = index

    def forward(self, ctx, x):  # x: [B, N]
        return x[:, self.index]  # [B]