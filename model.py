import os
import torch
import torch.nn as nn
from utils import Action

class DQN(nn.Module):
    def __init__(self, n_action, channel_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.channel_dim = channel_dim
        
        self.conv1 = nn.Conv2d(1, self.channel_dim//2, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(self.channel_dim//2, self.channel_dim//2, kernel_size=3, padding='same')

        self.conv3 = nn.Conv2d(self.channel_dim//2, self.channel_dim, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(self.channel_dim, self.channel_dim, kernel_size=3, padding='same')

        self.lazylinear = nn.LazyLinear(self.channel_dim)
        self.linear1 = nn.Linear(self.channel_dim, self.n_action)
        self.linear2 = nn.Linear(self.channel_dim, self.n_action)

        self.avgpool = nn.AvgPool2d(2, 2)

        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.batchnorm1 = nn.BatchNorm2d(self.channel_dim//2, eps=layernorm_eps)
        self.batchnorm2 = nn.BatchNorm2d(self.channel_dim, eps=layernorm_eps)

        self.flatten = nn.Flatten()

        self.drop = nn.Dropout(p=dropout_rate)
    
    def forward(self, vision):
        vision = self.relu(self.batchnorm1(self.conv1(vision)))
        vision = self.relu(self.batchnorm1(self.conv2(vision)))
        vision = self.avgpool(vision)

        vision = self.drop(vision)

        vision = self.relu(self.batchnorm2(self.conv3(vision)))
        vision = self.relu(self.batchnorm2(self.conv4(vision)))
        vision = self.avgpool(vision)

        vision = self.drop(vision)

        vision = self.flatten(vision)
        vision = self.relu(self.lazylinear(vision))
        
        vision = self.drop(vision)
        vision = self.linear1(vision)

        return vision

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DQTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, cur_state, reward, action, next_state, agent_dead):
        cur_state = cur_state.clone().detach()
        next_state = next_state.clone().detach()
        action = action.clone().detach()
        reward = reward.clone().detach()

        if len(action.shape) == 1:
            cur_state = torch.unsqueeze(cur_state, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            agent_dead = (agent_dead, )
        
        cur_state = torch.unsqueeze(cur_state, dim=1)
        next_state = torch.unsqueeze(next_state, dim=1)
        
        # compute predicted Q value with 
        pred = self.model(cur_state.to(self.device))

        # reward + gamme * max(next_pred Q value)
        target = pred.clone()
        # target = torch.zeros_like(pred)

        new_pred = self.model(next_state.to(self.device))
        for idx in range(len(agent_dead)):
            Q_new = reward[idx].to(self.device)
            if not agent_dead[idx]:
                Q_new += self.gamma * torch.max(new_pred[idx])
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        # Compute Loss and Optimize
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()