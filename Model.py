import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.common_layer = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.common_layer(state))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value
