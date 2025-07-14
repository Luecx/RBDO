from ..core import Node, IndexNode
import torch

class Realization(Node):
    def __init__(self, z_node, corr):
        super().__init__([z_node])
        self.random_vars = corr.random_vars

    def forward(self, ctx, z):  # z: [B, N]
        x_list = []
        for i, rv in enumerate(self.random_vars):
            z_i = z[:, i]  # [B]
            x_i = rv.sample(z_i)  # [B]
            x_list.append(x_i.unsqueeze(1))  # [B,1]
        return torch.cat(x_list, dim=1)  # [B, N]

    def __getitem__(self, rv):
        return IndexNode(rv._id, self)

