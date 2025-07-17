from enum import Enum, auto

class NodeKind(Enum):
    STANDARD = auto()  # regular node
    U        = auto()  # returns u
    V        = auto()  # returns v


from enum import Enum, auto

class DerivativeMode(Enum):
    ANALYTIC = auto()  # supports autograd
    NUMERIC  = auto()  # uses numerical differentiation
