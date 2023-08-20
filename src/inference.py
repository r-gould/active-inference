import math
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float
from torch.distributions import Normal

pi = math.pi
e = math.e

def inference(
    state: Float[Tensor, "s"], 
    transition,#: Transition, 
    reward,#: Reward,
    num_ensembles, action_dim, H, J,
    num_opt_iters, take_top, device,
    r_max, r_multiplier, alpha
) -> Float[Tensor, "a"]:

    q_mean = torch.zeros(H, J, action_dim).to(device)
    q_std = torch.ones(H, J, action_dim).to(device)
    state_dim = state.shape[-1]
    # (n, J, S)
    state = state.unsqueeze(0).unsqueeze(0).repeat(num_ensembles, J, 1)

    for i in range(num_opt_iters):

        actions = (q_mean + q_std * torch.randn_like(q_mean).to(device)
                   ).unsqueeze(0).repeat(num_ensembles, 1, 1, 1)
        
        energies = torch.zeros(J).to(device)

        states = [state] + [None for _ in range(H)]

        min_variance = 1e-7

        for t in range(H):
            curr_state = states[t] # (n, J, S)
            curr_action = actions[:, t, :, :] # (n, J, A)
            trans_mean, trans_var = transition(curr_state, curr_action) # each (n, J, S)
            
            next_state = curr_state + Normal(trans_mean, torch.sqrt(trans_var)).rsample() # (n, J, S)
            states[t+1] = next_state

            curr_energy = 0

            rew_mean, rew_var = reward(curr_state, curr_action, sample=False) # (n, J)

            curr_energy += 0.5 * (((rew_mean - r_max*r_multiplier)**2 + rew_var) / (alpha**2) 
                                  - torch.log(torch.clamp(rew_var, min=min_variance) / alpha**2) - 1) # (n, J)

            curr_energy += -0.5 * torch.sum(
                torch.log(2*pi*e*torch.clamp(trans_var, min=min_variance)), dim=-1
            ) # (n, J)

            energies += torch.mean(curr_energy, dim=0)
        
        _, topk = energies.topk(take_top, dim=0, largest=False, sorted=False)
        best_actions = actions[0, :, topk.view(-1)].reshape(
            H, take_top, action_dim
        )
        # each (H, 1, A)
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )

        q_mean = action_mean.repeat(1, J, 1)
        q_std = action_std_dev.repeat(1, J, 1)

    return q_mean[0, 0, :]