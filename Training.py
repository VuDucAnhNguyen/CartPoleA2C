import gymnasium as gym
import torch
from Params import params  # Import Params instance
from Utils import utils

class Training():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        torch.manual_seed(params.seed)
        
    def start_training(self):
        for episode in range(1, params.training_num_episodes + 1):
            state, _ = self.env.reset(seed=params.seed)  # ✅ seed môi trường và unpack info
            log_probs = []
            state_values = []
            rewards = []
            masks = []

            total_reward = 0
            done = False

            while not done:
                action, log_prob, state_value = self.agent.get_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)  # ✅ Gymnasium
                done = terminated or truncated

                log_probs.append(log_prob)
                state_values.append(state_value)
                rewards.append(reward)
                masks.append(1 - int(done))

                state = next_state
                total_reward += reward

            loss = self.agent.compute_loss(log_probs, state_values, rewards, masks)
            self.agent.update_model(loss)

            if episode % 100 == 0:
                print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Loss: {loss.item():.4f}")

        self.agent.save_model(params.save_path)
        print(f"Training completed. Model saved to {params.save_path}")

