import torch
import torch.nn as nn

from typing import List, Callable
from torch import Tensor
from jaxtyping import Float

def swish(x: Tensor):
    return x * torch.sigmoid(x)

class EnsembleLayer(nn.Module):

    def __init__(self, num_ensembles: int, input_dim: int, output_dim: int,
                 act_fn):#nn.ReLU):
        super().__init__()

        self.W = nn.Parameter(torch.zeros((num_ensembles, input_dim, output_dim)))
        self.b = nn.Parameter(torch.zeros((num_ensembles, 1, output_dim)))

        for w in self.W:
            if act_fn == "linear":
                nn.init.xavier_normal_(w)
            else:
                nn.init.xavier_uniform_(w)

        if act_fn == "swish":
            self.act_fn = swish
        elif act_fn == "linear":
            self.act_fn = (lambda x: x)
        else:
            raise ValueError()

    def forward(
        self, x: Float[Tensor, "n b input"]
    ) -> Float[Tensor, "n b output"]:
        
        out = self.act_fn(torch.bmm(x, self.W) + self.b)
        return out

class Ensemble(nn.Module):
    """
    A model made up of multiple EnsembleLayers.
    """

    def __init__(self, num_ensembles: int, node_info: List[int], act_fn):
    #def __init__(self, num_ensembles: int, act_fn, in_size, hidden_size, out_size):
        super().__init__()

        self.model = nn.Sequential(
            *[EnsembleLayer(num_ensembles, node_info[i], node_info[i+1], 
                            act_fn if i < len(node_info)-2 else "linear")
              for i in range(len(node_info)-1)]
        )

    def forward(
        self, x: Float[Tensor, "n b input"]
    ) -> Float[Tensor, "n b output"]:

        return self.model(x)