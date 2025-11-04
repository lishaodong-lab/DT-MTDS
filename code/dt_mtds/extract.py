import os
import argparse
import torch
import sys
import torch.nn as nn
sys.path.append("/home/tjx/ManiSkill-ViTac2024-main")


from envs.env_for_stack import ur5Env_1
from pcrl.utils import collect_demo
from pcrl.network import StateIndependentPolicy
from gym import spaces

from tqdm import tqdm
import numpy as np
import pickle
from pcrl.buffer import Buffer



class Transition:
    def __init__(self, state, action):
        self.state = state
        self.action = action


state_space = spaces.Box(
            low=np.array(
                [0.48 * 100, -0.15 * 100, 0.0,
                 0.48 * 100, -0.15 * 100, 0.0
                ],
                dtype=float),
            high=np.array(
                [0.72 * 100, 0.15 * 100, 0.3 * 100,
                 0.72 * 100, 0.15 * 100, 0.3 * 100
                 ]),
            dtype=float
        )
action_space = spaces.Box(low=np.array(
            [-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)
# Actor.
actor = StateIndependentPolicy(
            state_shape=state_space.shape,
            action_shape=action_space.shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
                               )





def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0
    success_num = 0
    transitions = []


    state = env.reset()
    t = 0
    episode_return = 0.0

    state_shape = 6
    file_path = '/home/tjx/logs/Hopper-v3/sac/seed0-20241213-1611/model/step50000/actor.pth'
    # 加载模型参数
    model1_state_dict = torch.load(file_path)
    actor.load_state_dict(model1_state_dict)

    for _ in tqdm(range(1, 10**4 + 1)):

        state_tensor = torch.from_numpy(state).clone().float()
        output = actor(state_tensor)
        print(output)
        action_tensor = actor(state_tensor)
        state = torch.tensor(state, dtype=torch.float, device="cpu")
        with torch.no_grad():
            action = actor(state.unsqueeze_(0))
        action = action.cpu().numpy()[0]

        next_state, reward, done, _ = env.step(action)
 
        mask = False if t == env._max_episode_steps else done

           
        buffer.append(state, action, reward, mask, next_state)
        state = next_state
        # print(reward)
        episode_return += reward

        if done:
           num_episodes += 1
           total_return += episode_return
           state = env.reset()
           t = 0
           episode_return = 0.0

        if reward == 5000:
            with open(file_path1, 'rb') as file:
             permanent_data = pickle.load(file)

            with open(file_path0, 'rb') as file:
             transient_data = pickle.load(file)

            permanent_data.append(transient_data)
            with open(file_path1, 'wb') as file:
              pickle.dump(permanent_data, file)
            transitions = []
            success_num +=1
        else:
            with open(file_path0, 'wb') as file:
              pickle.dump([], file)
            transitions = []
         



    print(f'Mean return of the expert is {total_return / num_episodes}')
    print(success_num)
    print(num_episodes)
    print(total_return)
    success_rate = success_num / num_episodes
    print(f"Success rate: {success_rate:.2f}")
    return buffer



