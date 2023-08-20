import torch
import torch.nn as nn

from typing import List, Tuple
from torch import Tensor
from jaxtyping import Float
from torch.distributions import Normal

from .ensemble import Ensemble

def swish(x):
    return x * torch.sigmoid(x)

class Transition(nn.Module):
    
    #Models q(s_t | s_{t-1}, \theta, \pi), though with the caveeat that 
    #\delta s_t is predicted where s_t = s_{t-1} + \delta s_t
    

    def __init__(self, num_ensembles: int, node_info: List[int], normalizer, act_fn = "swish"):
        super().__init__()

        self.num_ensembles = num_ensembles
        self.node_info = node_info
        self.act_fn = act_fn
        self.normalizer = normalizer

        self.model = Ensemble(num_ensembles, node_info, act_fn)

        output_dim = node_info[-1]
        assert((output_dim % 2) == 0)
        self.split_at = output_dim // 2

        self.max_logvar = -1
        self.min_logvar = -5

    def forward(
        self, state: Float[Tensor, "n b s"],
        action: Float[Tensor, "n b a"], denorm=True
    ) -> Tuple[Float[Tensor, "n b s"], Float[Tensor, "n b s"]]:
        
        state = self.normalizer.normalize_states(state)
        action = self.normalizer.normalize_actions(action)

        state_action = torch.cat((state, action), dim=-1)

        mean_logvar = self.model.forward(state_action)
        mean, logvar = mean_logvar[:, :, :self.split_at], mean_logvar[:, :, self.split_at:]
        logvar = torch.sigmoid(logvar)
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * logvar
        
        var = torch.exp(logvar)

        if denorm is True:
            mean = self.normalizer.denormalize_state_delta_means(mean)
            var = self.normalizer.denormalize_state_delta_vars(var)

        return (mean, var)

    def loss(
        self, states: Float[Tensor, "n b s"], actions: Float[Tensor, "n b a"],
        delta_states#next_states: Float[Tensor, "n b s"]
    ) -> Float[Tensor, ""]:
        
        #states = self.normalizer.normalize_states(states)
        #actions = self.normalizer.normalize_actions(actions)

        true = delta_states #next_states - states
        true = self.normalizer.normalize_state_deltas(true)


        mean, var = self.forward(states, actions, denorm=False)
        #neg_log_prob = -Normal(mean, torch.sqrt(var)).log_prob(true).mean(-1).mean(-1).sum()
        
        #loss = (mean - true) ** 2 / var + torch.log(var)
        loss = -Normal(mean, torch.sqrt(var)).log_prob(true)
        loss = loss.mean(-1).mean(-1).sum()
        return loss
    
    def reset(self):
        del self.model
        self.model = Ensemble(self.num_ensembles, self.node_info, self.act_fn)

    def sample(self, means, vars):
        return torch.distributions.Normal(means, torch.sqrt(vars)).rsample()