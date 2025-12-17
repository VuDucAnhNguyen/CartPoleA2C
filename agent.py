import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic
from params import params

class A2CAgent:
    def __init__(self, input_dim, n_actions):
        self.gamma = params.gamma
        self.lr = params.learning_rate
        self.model = ActorCritic(input_dim, n_actions).to(params.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(params.device)
        action_probs, state_value = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, state_value, entropy

    def compute_loss(self, log_probs, state_values, rewards, masks, entropies, next_state_value):
        returns = []
        R = next_state_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(params.device)
        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values).view(-1)
        entropy_loss = torch.stack(entropies).mean()
        
        advantage = returns - state_values
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = torch.nn.functional.mse_loss(state_values, returns)
        loss = actor_loss + critic_loss - params.beta * entropy_loss

        return loss

    def update_model(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
