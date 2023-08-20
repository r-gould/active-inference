import random
from collections import deque
import os

from tqdm import tqdm
import numpy as np
import torch

from src import Dataset, Transition, Reward, Normalizer
from env import GymEnv

from copy import deepcopy

def main(env_name, use_orig=False):

    if env_name in ["HalfCheetahRun", "HalfCheetahFlip"]:
        action_repeat = 2
        max_ep_len = 100
        env = GymEnv(env_name, max_ep_len, action_repeat)

        num_seed_episodes = 5
        batch_size = 50
        num_episodes = 100
        epochs = 100
        hidden_size = 400
        lr = 0.001
        eps = 1e-8
        store_len = 100
        take_top = 70
        J = 700
        num_opt_iters = 7
        H = 15
        num_ensembles = 16
        grad_clip_norm = 1000
        expl_scale = 0.1

    elif env_name == "AntMaze":
        action_repeat = 4
        max_ep_len = 300
        env = GymEnv(env_name, max_ep_len, action_repeat)

        num_seed_episodes = 5
        batch_size = 50
        num_episodes = 50
        epochs = 100
        hidden_size = 400
        lr = 0.001
        eps = 1e-8
        store_len = 100
        take_top = 70
        J = 700
        num_opt_iters = 7
        H = 30
        num_ensembles = 15
        grad_clip_norm = 1000
        expl_scale = 1.0
    else:
        ValueError()


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = "cuda"

    
    normalizer = Normalizer()
    transition = Transition(num_ensembles, 
                            node_info=[state_dim+action_dim, hidden_size, hidden_size, hidden_size, 2*state_dim],
                            normalizer=normalizer)
    transition.to(device)

    reward = Reward(node_info=[state_dim+action_dim, hidden_size, hidden_size, 1])
    reward.to(device)

    r_max = -1e6

    actor = None
    if use_orig:
        print("Using Tschantz et al.")
        
        from src.orig import Planner

        planner = Planner(transition,
            reward,
            action_dim,
            num_ensembles,
            H,
            num_opt_iters,
            J,
            take_top,
            use_reward=True,
            use_exploration=True,
            use_mean=True,
            expl_scale=expl_scale,
            strategy="information",
            device=device)

        actor = planner.forward

    else:
        from src import inference

        r_multiplier = 1.5
        alpha = 0.5**(0.5)

        actor = lambda state: inference(
            torch.from_numpy(state).float().to(device), transition, reward, num_ensembles, 
            action_dim, H, J, num_opt_iters, take_top, 
            device, r_max, r_multiplier, alpha
        )

    dataset = Dataset(num_ensembles, state_dim, action_dim, batch_size, device, normalizer)
    entire_reward_store = []
    episode_rewards = deque(maxlen=store_len)
    
    for _ in range(num_seed_episodes):

        state = env.reset()
        done = False
        count = 0

        while (not done):
            action = env.action_space.sample()
            next_state, rew, done, _ = env.step(action)
            dataset.add(state, action, next_state, rew)
            r_max = max(r_max, rew)

            state = deepcopy(next_state)
            count += 1

            if count >= max_ep_len:
                break

    print("Collected seed episodes")

    for ep in tqdm(range(num_episodes+5)):
        print("Episode", ep)

        transition.reset()
        reward.reset()
        
        transition.to(device)
        reward.to(device)


        params = list(transition.parameters()) + list(reward.parameters())
        opt = torch.optim.Adam(
            params, lr=lr, eps=eps
        )

        losses = []
        trans_losses = []
        rew_losses = []

        for epoch in range(epochs):
            
            for (states, actions, next_states, rewards) in dataset:

                transition.train()
                reward.train()

                transition_loss = transition.loss(states, 
                    actions, next_states - states)
                reward_loss = reward.loss(states, actions, rewards)
                loss = transition_loss + reward_loss
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params, grad_clip_norm, norm_type=2
                )
                opt.step()

                losses.append(loss.item())
                trans_losses.append(transition_loss.item())
                rew_losses.append(reward_loss.item())

        print(f"Avg trans loss: {np.mean(trans_losses)}, avg rew loss: {np.mean(rew_losses)}")

        ep_reward = 0


        state = env.reset()
        done = False

        count = 0

        indiv_rews = []

        while (not done):
            with torch.no_grad():
                action = actor(state).detach().cpu().numpy()
            next_state, rew, done, _ = env.step(action)
            r_max = max(r_max, rew)
            

            dataset.add(state, action, next_state, rew)

            state = deepcopy(next_state)
            
            indiv_rews.append(rew)
            ep_reward += rew
            count += 1


            if count >= max_ep_len:
                break

        episode_rewards.append(ep_reward)
        entire_reward_store.append(ep_reward)

        print(f"Reward:", ep_reward)
        print("r_max:", r_max)

        if ((ep % 25) == 0):
            np.save(f"saved/all_rewards_{ep}.npy", entire_reward_store)

        print()

if __name__ == '__main__':
    env_name = "HalfCheetahFlip"
    main(env_name, use_orig=False)