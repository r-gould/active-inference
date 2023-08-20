import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
from torch import Tensor
from jaxtyping import Float

#from ensemble import EnsembleLayer

class Reward(nn.Module):
    """
    Models q(r_t | s_t, \theta, \pi), where r_t is the reward at 
    timestep t.
    """

    def __init__(self, node_info: List[int], act_fn = nn.ReLU):
        super().__init__()
        
        self.node_info = node_info
        self.act_fn = act_fn

        assert(len(node_info) == 4)
        assert(node_info[-1] == 1)
        self.model = nn.Sequential(
            nn.Linear(node_info[0], node_info[1]),
            nn.ReLU(),
            nn.Linear(node_info[1], node_info[2]),
            nn.ReLU(),
            nn.Linear(node_info[2], 2*node_info[3])
        )

        self.max_logvar = -1
        self.min_logvar = -5
        

    def forward(
        self, states: Float[Tensor, "*b s"], actions: Float[Tensor, "*b a"], sample=True
    ):# -> Float[Tensor, "*b"]:

        states_actions: Float[Tensor, "*b s+a"]
        states_actions = torch.cat((states, actions), dim=-1)

        mean_logvar = self.model.forward(states_actions) # (*b 2)
        assert(mean_logvar.shape[-1] == 2)
        mean, logvar = mean_logvar[..., 0], mean_logvar[..., 1]
        logvar = torch.sigmoid(logvar)
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * logvar
        
        var = torch.exp(logvar)

        if sample:
            rewards = torch.distributions.Normal(mean, torch.sqrt(var)).rsample()
            return rewards
        
        return mean, var

    def loss(
        self, states: Float[Tensor, "n b s"], actions: Float[Tensor, "n b a"],
        rewards: Float[Tensor, "n b 1"]
    ) -> Float[Tensor, ""]:
        
        pred_rewards = self.forward(states, actions) # (n, b)
        error = F.mse_loss(pred_rewards, rewards.squeeze())
        return error

    def reset(self):
        del self.model

        self.model = nn.Sequential(
            nn.Linear(self.node_info[0], self.node_info[1]),
            nn.ReLU(),
            nn.Linear(self.node_info[1], self.node_info[2]),
            nn.ReLU(),
            nn.Linear(self.node_info[2], 2*self.node_info[3])
        )