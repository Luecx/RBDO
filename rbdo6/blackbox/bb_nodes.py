from .bb_func import *
from ..core import Node

class BlackBoxNode(Node):
    def __init__(self, fn, input_nodes):
        super().__init__(input_nodes)
        self.fn = fn
        self._uses_numerical = True

    def forward(self, ctx, *args):
        return BlackBoxFunction.apply(self.fn, *args)
