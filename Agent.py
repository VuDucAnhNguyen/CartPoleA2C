import torch
import torch.optim as optim
import numpy as np
from Model import ActorCritic
from Params import params

class Agent:
    def __init__(self, input_dim, n_actions):
        self.gamma = params.gamma
        self.lr = params.learning_rate
        self.model = ActorCritic(input_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs, state_value = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, state_value

    def compute_loss(self, log_probs, state_values, rewards, masks):
        returns = []
        R = 0
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values).squeeze()
        
        advantage = returns - state_values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = torch.nn.functional.mse_loss(state_values, returns)
        loss = actor_loss + critic_loss
        return loss

    def update_model(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()