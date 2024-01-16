import os
import torch
import torch.nn as nn
from utils import Action

class DQN(nn.Module):
    def __init__(
        self, 
        n_action :int,
        channel_dim :int, 
        channel_dim_out :int, 
        dropout_rate :float = 0.1, 
        layernorm_eps :float = 1e-6
        ) -> None:
        super(DQN, self).__init__()

        self.n_action = n_action
        self.channel_dim = channel_dim
        self.channel_dim_out = channel_dim_out
        
        self.convstart1 = nn.Conv2d(self.channel_dim, self.channel_dim_out//2, kernel_size=3)
        self.convstart2 = nn.Conv2d(self.channel_dim_out//2, self.channel_dim_out, kernel_size=3, padding='same')

        self.conv1 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')

        self.conv3 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')

        self.conv5 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')
        self.conv6 = nn.Conv2d(self.channel_dim_out, self.channel_dim_out, kernel_size=3, padding='same')

        self.lazyfc = nn.LazyLinear(self.channel_dim_out)
        self.fc = nn.Linear(self.channel_dim_out, self.n_action)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.bnstart1 = nn.BatchNorm2d(self.channel_dim_out//2, eps=layernorm_eps)
        self.bnstart2 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn1 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn2 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn3 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn4 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn5 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)
        self.bn6 = nn.BatchNorm2d(self.channel_dim_out, eps=layernorm_eps)

        self.flatten = nn.Flatten()

        self.drop = nn.Dropout(p=dropout_rate)
    
    def forward(self, vision :torch.Tensor) -> torch.Tensor:
        
        # First setup Conv
        x = self.convstart1(vision)
        x = self.bnstart1(x)
        x = self.relu(x)
        x = self.convstart2(x)
        x = self.bnstart2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Simple Resnet part
        
        # Block 1
        vision1 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += vision1
        x = self.relu(x)

        # Block 2
        vision2 = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x += vision2
        x = self.relu(x)

        # Block 3
        vision3 = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x += vision3
        x = self.relu(x)

        # Feed Forward part
        x = self.avgpool(x)

        x = self.flatten(x)
        x = self.relu(self.lazyfc(x))
        
        x = self.fc(x)

        return x

    def save(self, file_name :str = 'model.pth') -> None:
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DQTrainer:
    def __init__(
        self, model :nn.Module, 
        lr :float, 
        gamma :float, 
        device :torch.device
        ) -> None:

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    # Traingin step using the DQN scheme
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
        
        # cur_state = torch.unsqueeze(cur_state, dim=1)
        # next_state = torch.unsqueeze(next_state, dim=1)
        
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