import os
import sys
import torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('/MTDS/run_loss')

states_data_list =[]
actions_data_list = []
num = 0

with open('/home/tjx/teacher_datas_peg_in_hole/transfer', 'rb') as file:
    loaded_data0 = pickle.load(file)

with open('/home/tjx/eacher_datas_push_reach/transfer', 'rb') as file:
    loaded_data1 = pickle.load(file)

"..."


for transition in loaded_data0:

    state = transition.state
    state_tensor = torch.from_numpy(state).clone().float()


    flag_tensor = torch.tensor([0], dtype=torch.float32) 


    state_tensor = torch.cat((state_tensor, flag_tensor), dim=0)

    states_data_list.append(state_tensor)


    action = transition.action
    action_tensor = torch.from_numpy(action).clone()

    actions_data_list.append(action_tensor)
    num=num+1
print(num)

for transition in loaded_data1:

    state = transition.state
    state_tensor = torch.from_numpy(state).clone().float()

    flag_tensor = torch.tensor([0], dtype=torch.float32)  

    state_tensor = torch.cat((state_tensor, flag_tensor), dim=0)

    states_data_list.append(state_tensor)

    action = transition.action
    action_tensor = torch.from_numpy(action).clone()

    actions_data_list.append(action_tensor)
    num=num+1
print(num)

".."

states_tensor = torch.stack(states_data_list)
actions_tensor = torch.stack(actions_data_list)

dataset = TensorDataset(states_tensor, actions_tensor)


batch_size = 32
dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(in_features = 7, out_features = 64)
        self.layer2 = nn.Linear(in_features = 64, out_features = 32)
        self.layer3 = nn.Linear(in_features = 32, out_features = 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

num_epochs =2000


model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

loss_history = []


for epoch in range(num_epochs):
    
    for batch_states, batch_actions in dataloader:

        output = model(batch_states)

        

        loss = criterion(output, batch_actions)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch)
        loss_history.append(loss.item())

writer.close()

print("distillation has already down")
directory = '/home/tjx/ManiSkill-ViTac2024-main/distillation'
file_name = 'find_into_two_cubes_exchange_distillation_pth'
file_path = os.path.join(directory, file_name)

if not os.path.exists(file_path):
    open(file_path, 'w').close()


model_state_dict = model.state_dict()

torch.save(model_state_dict, file_path)
