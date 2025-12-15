import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from Params import Params
class Training:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def run_env(self):
        self.agent.model.train()
        params = Params() 
        n_episode = params.training_num_episodes
        raw_history = []
        smoothed_history = []
        running_reward = 0
        for episode in range(n_episode):
            done = False
            log_probs = []
            values = []
            rewards = []
            masks = []
            state, info = self.env.reset()
            total_rewards = 0
            step = 0
            while not done and step <= params.max_steps:
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = truncated or terminated
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(0 if done else 1)
                total_rewards = total_rewards + reward
                state = next_state
                step = step + 1
            loss = self.agent.compute_loss(log_probs, values, rewards, masks)
            self.agent.update_model(loss)
            if episode == 0:
               running_reward = total_rewards
            else:
                running_reward = 0.05 * total_rewards + 0.95 * running_reward
            raw_history.append(total_rewards)
            smoothed_history.append(running_reward)
            if episode % 10 == 0:
                print(f"Episode {episode} \t Raw {total_rewards} \t Smooth {running_reward}" )
        self.agent.save_model("actor_critic.pth")
        self.plot_learning_curve(raw_history, smoothed_history)
        self.env.close()
    def plot_learning_curve(self, raw_rewards, smoothed_rewards):
        plt.figure(figsize=(10, 6))
        plt.plot(raw_rewards, label='Raw Reward (Episode)', color='cyan', alpha=0.3)
        plt.plot(smoothed_rewards, label='Smoothed Reward (Trend)', color='orange', linewidth=2)
        plt.title("Training Learning Curve (Actor-Critic)")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Total Reward per Episode")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_curve.png")
        plt.show()
    if __name__ == "main":
        run_env()

            


                
