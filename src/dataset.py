import torch
import numpy as np

from typing import List, Callable
from torch import Tensor
from jaxtyping import Float

from numpy.random import default_rng

class Dataset:
    def __init__(self, num_ensembles, state_dim, action_dim, batch_size, device, normalizer, 
                capacity=int(1e6)):
        self.ptr = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_ensembles = num_ensembles
        self.device = device
        self.normalizer = normalizer

        self.states = np.empty((capacity, state_dim))
        self.actions = np.empty((capacity, action_dim))
        self.next_states = np.empty((capacity, state_dim))
        self.rewards = np.empty((capacity, 1))

    def add(
        self, state: Float[np.ndarray, "s"], action: Float[np.ndarray, "a"],
        next_state: Float[np.ndarray, "s"], reward: Float[np.ndarray, "1"]
    ):

        self.states[self.ptr, :] = state
        self.actions[self.ptr, :] = action
        self.next_states[self.ptr, :] = next_state
        self.rewards[self.ptr, :] = reward
        
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.ptr = self.ptr % self.capacity

        self.normalizer.update(state, action, next_state - state)

    def __iter__(self):
        idxs = torch.cat([torch.randperm(self.ptr).unsqueeze(0) for _ in range(self.num_ensembles)],
                         dim=0) # (num_ensembles, ptr)
        for i in range(0, self.ptr, self.batch_size):
            curr_idxs = idxs[:, i : i+self.batch_size].unsqueeze(-1) # (num_ensembles, batch_size, 1)
            
            states = torch.from_numpy(self.states)[curr_idxs].squeeze(2).float().to(self.device) # (n, b, s)
            actions = torch.from_numpy(self.actions)[curr_idxs].squeeze(2).float().to(self.device)
            next_states = torch.from_numpy(self.next_states)[curr_idxs].squeeze(2).float().to(self.device)
            rewards = torch.from_numpy(self.rewards)[curr_idxs].squeeze(2).float().to(self.device)

            yield (states, actions, next_states, rewards)