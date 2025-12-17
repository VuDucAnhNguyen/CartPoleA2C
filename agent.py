import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from model import ActorCritic
from params import params

class A2CAgent:
    def __init__(self, input_dim, n_actions):
        self.gamma = params.gamma
        self.entropy_beta = params.entropy_beta
        self.device = params.device

        self.model = ActorCritic(input_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate)

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs, value = self.model(state)

        if deterministic:
            action = torch.argmax(probs)
            log_prob = torch.log(probs[action] + 1e-8)
            entropy = torch.tensor(0.0).to(self.device)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob, entropy, value

    def compute_n_step_returns(self, rewards, masks, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return torch.stack(returns)

    def update(self, log_probs, values, rewards, masks, entropies, next_state):
        with torch.no_grad():
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            _, next_value = self.model(next_state)

        returns = self.compute_n_step_returns(rewards, masks, next_value)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze(-1)
        entropies = torch.stack(entropies)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, returns.squeeze(-1))
        entropy_loss = entropies.mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load_model(self, path, eval_mode=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if eval_mode:
            self.model.eval()
