import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np
from .base import Algorithm
from pcrl.buffer import RolloutBuffer, RolloutBuffer1, RolloutBuffer2
from pcrl.network import StateIndependentPolicy, StateFunction, StateIndependentPolicy1
from pcrl.utils import disable_gradient


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.loss_critic = (self.critic(states) - targets).pow_(2).mean()
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class gating_func(nn.Module):
    def __init__(self, num_inputs, hidden_dim=256, K_primitives=3):
        super(gating_func, self).__init__()
        self.NN_w = nn.Sequential(nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, K_primitives),
            nn.Softmax())

    def forward(self, state):
        return self.NN_w(state)


class PPO(Algorithm):

    def __init__(self, state_shape,  action_shape,  device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(256, 256),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,          # coef_ent = 0.01
                 max_grad_norm=10.0):
        super().__init__(state_shape,  action_shape, device, seed, gamma)

        # Rollout buffer.
        self.rollout_length = rollout_length
        self.x = 0
        self.y = 0
        self.buffer1 = RolloutBuffer1(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer              # 这里的mix还是1
        )           


        self.buffer2 = RolloutBuffer2(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        self.buffer3 = RolloutBuffer2(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        self.buffer4 = RolloutBuffer2(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )


        # Actor.
        self.actor1 = StateIndependentPolicy1(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.actor2 = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(device)



        # 这里输出的就是action对应的mean




        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)




        self.optim_actor1 = Adam(self.actor1.parameters(), lr=lr_actor, eps=1e-5)
        self.optim_actor2 = Adam(self.actor2.parameters(), lr=lr_actor, eps=1e-5)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.step_num = 0


    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):

        t += 1
        self.step_num +=1

        info = env.infomation()
        if self.step_num < 2000:

         if info == 1:
            # state1 = np.append(state[:6], 0.0)
            action1, log_pi_1 = self.explore1(state) # 要修改explore函数
            next_state, reward, done, reward_c = env.step(action1)
            self.buffer1.append(state, action1, reward, done, log_pi_1, next_state)
            self.buffer3.append(state, action1, reward, done, log_pi_1, next_state)
            self.x +=1

         elif info == 0:
            # state2 = np.append(state[:6], 1.0)
            action2, log_pi_2 = self.explore2(state)
            change = 0
            next_state, reward2, done, _ = env.step(action2)
            self.buffer2.append(state, action2, reward2, done, log_pi_2, next_state)
            self.buffer3.append(state, action2, reward2, done, log_pi_2, next_state)
            self.y +=1


        



         if done:
            t = 0
            next_state = env.reset()


        if self.step_num == 2000:

            next_state = env.reset()

        if 2000 < self.step_num <= 6000:

         action, log_pi = self.explore(state)
         change = 0
         next_state, reward, done, _ = env.step(action)
         self.buffer4.append(state, action, reward, done, log_pi, next_state)
         self.buffer3.append(state, action, reward, done, log_pi, next_state)
         print(self.step_num)
        #  print(next_state)

         if done:
            t = 0
            next_state = env.reset()

        if self.step_num > 6000:

            self.step_num = 0
            next_state = env.reset()

        # print(self.step_num)

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states1, actions1, rewards1, dones1, log_pis1, next_states1 = \
            self.buffer1.get()

        self.update_ppo1(
            states1, actions1, rewards1, dones1, log_pis1, next_states1, writer)

        if self.buffer2._n2 >= self.rollout_length - 1:
            states2, actions2, rewards2, dones2, log_pis2, next_states2 = \
                self.buffer2.get()
            self.update_ppo2(
                states2, actions2, rewards2, dones2, log_pis2, next_states2, writer)
            
            
        if self.buffer3._n2 >= self.rollout_length - 1:
            states3, actions3, rewards3, dones3, log_pis3, next_states3 = \
                self.buffer3.get()
            self.update_ppo3(
                states3, actions3, rewards3, dones3, log_pis3, next_states3, writer)
            
        if self.buffer4._n2 >= self.rollout_length - 1:
            states4, actions4, rewards4, dones4, log_pis4, next_states4 = \
                self.buffer4.get()
            self.update_ppo4(
                states4, actions4, rewards4, dones4, log_pis4, next_states4, writer)
            
            
        



    def update_ppo1(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            # self.update_critic(states, targets, writer)
            self.update_actor1(states, actions, log_pis, gaes, writer)

    def update_ppo2(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            # self.update_critic(states, targets, writer)
            self.update_actor2(states, actions, log_pis, gaes, writer)



    def update_ppo3(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)


    def update_ppo4(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)




    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor1(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor1.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor1 = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor1.zero_grad()
        (loss_actor1 - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor1.parameters(), self.max_grad_norm)
        self.optim_actor1.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor1', loss_actor1.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy1', entropy.item(), self.learning_steps)

    def update_actor2(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor2.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor1 = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor2.zero_grad()
        (loss_actor1 - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor2.parameters(), self.max_grad_norm)
        self.optim_actor2.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor2', loss_actor1.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy2', entropy.item(), self.learning_steps)


    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor1 = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor1 - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor1.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)
            


            

    
            
    

    # def exploit1(self, state, info):
    #     state = torch.tensor(state, dtype=torch.float, device=self.device)
    #     if info:
    #         with torch.no_grad():
    #             action = self.actor1(state.unsqueeze_(0))
    #     else:
    #         with torch.no_grad():
    #             action = self.actor(state.unsqueeze_(0))
    #     return action.cpu().numpy()[0]

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor1.state_dict(),
            os.path.join(save_dir, 'actor1.pth')
        )
        torch.save(
            self.actor2.state_dict(),
            os.path.join(save_dir, 'actor2.pth')
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_dir, 'critic.pth')
        )
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


class PPOExpert(PPO):

    def __init__(self, state_shape, action_shape,  device, path1, path2, path3, path4, path5, path6, path7, path8):


        self.actor1 = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor1.load_state_dict(torch.load(path1))

        disable_gradient(self.actor1)
        self.device = device

        self.actor2 = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor2.load_state_dict(torch.load(path2))

        disable_gradient(self.actor2)
        self.device = device

