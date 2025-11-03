import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                #  hidden_units_r=(64, 64),
                #  hidden_units_v=(64, 64),
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(              # 用于生成当前状态的奖励
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(             # 用于估计当前状态和下一个状态的价值
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):    # 奖励计算函数f
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs         # rs是当前状态的奖励，vs是当前状态的价值，next_vs是下一个状态的价值，self.gamma * (1 - dones)计算折扣的未来价值

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis         # 输出是判别器的输出，表示对当前策略的评价(减去log_pis的操作在判别器的输出中引入了当前策略的选择概率，旨在调整奖励信号，使其更具有相对性和可用性)

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)                           # 利用判别器的输出计算奖励信号，使得agent能够通过判别器的评估来调整其策略



class AIRLSACDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                #  hidden_units_r=(64, 64),
                #  hidden_units_v=(64, 64),
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(              # 用于生成当前状态的奖励
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(             # 用于估计当前状态和下一个状态的价值
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):    # 奖励计算函数f
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs         # rs是当前状态的奖励，vs是当前状态的价值，next_vs是下一个状态的价值，self.gamma * (1 - dones)计算折扣的未来价值

    def forward(self, states, dones, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states)        # 输出是判别器的输出，表示对当前策略的评价(减去log_pis的操作在判别器的输出中引入了当前策略的选择概率，旨在调整奖励信号，使其更具有相对性和可用性)

    def calculate_reward(self, states, dones,next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, next_states)
            return -F.logsigmoid(-logits)          