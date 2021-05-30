from typing import List

import torch as th
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, net_arch: List[int], activation_fn: nn.Module = nn.PReLU, is_classification: bool = False
    ):
        super().__init__()
        layers = []
        for hidden_dim in net_arch[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim

        if len(net_arch) > 0:
            layers.append(nn.Linear(input_dim, net_arch[-1]))

        self.layers = nn.ModuleList(layers)
        self.is_classification = is_classification
        if is_classification:
            self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.is_classification:
            return self.softmax_layer(x)
        return x
