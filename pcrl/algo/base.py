from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]


    def explore1(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor1.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit1(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor1(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore2(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor2.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit2(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor2(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore3(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor3.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit3(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor3(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore4(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor4.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit4(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor4(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore5(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor5.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit5(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor5(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore6(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor6.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit6(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor6(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore7(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor7.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit7(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor7(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore8(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor8.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit8(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor8(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    
    def explore9(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor9.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit9(self, state):           # exploit在这里，使用模型actor来预测动作
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor9(state.unsqueeze_(0))
        return action.cpu().numpy()[0]
    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
