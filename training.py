import torch
from utils import utils
from params import params

class Training:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_training(self):
        self.agent.model.train()
        n_episode = params.training_num_episodes
        raw_history = []
        smoothed_history = []
        running_reward = 0

        for episode in range(n_episode):
            done = False
            log_probs, values, rewards, masks, entropies = [], [], [], [], []
            state, _ = self.env.reset()
            total_reward = 0

            while not done:
                action, log_prob, entropy, value = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                mask = 0.0 if done else 1.0

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.agent.device))
                masks.append(torch.tensor(mask, dtype=torch.float32, device=self.agent.device))
                entropies.append(entropy)

                state = next_state
                total_reward += reward

            # Update agent using n-step returns
            self.agent.update(log_probs, values, rewards, masks, entropies, next_state)

            # Update running reward
            if episode == 0:
                running_reward = total_reward
            else:
                running_reward = 0.05 * total_reward + 0.95 * running_reward

            raw_history.append(total_reward)
            smoothed_history.append(running_reward)

            if episode % 10 == 0:
                print(f"Episode {episode}\tRaw: {total_reward:.2f}\tSmooth: {running_reward:.2f}")

        utils.save_model(self.agent)
        utils.plot_learning_curve(raw_rewards=raw_history, smoothed_rewards=smoothed_history)
        self.env.close()
