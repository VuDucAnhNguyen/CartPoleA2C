import torch
import torch.nn as nn
import torch.nn.functional as F
from params import params

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.common_layer = nn.Linear(input_dim, params.hidden_dim)
        self.actor = nn.Linear(params.hidden_dim, n_actions)
        self.critic = nn.Linear(params.hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.common_layer(state))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value
