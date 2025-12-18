from params import params
from utils import utils
import torch

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

        for episode in range(1, n_episode + 1):
            done = False
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropies = []
            state, _ = self.env.reset()
            total_rewards = 0
            step = 0

            while not done:
                action, log_prob, value, entropy = self.agent.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = truncated or terminated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(0 if done else 1)
                entropies.append(entropy)

                total_rewards = total_rewards + reward
                state = next_state
                step += 1

                if (step % params.n_steps == 0 or done):
                    if done:
                        next_state_value = 0
                    else:
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(params.device)
                        _, next_value = self.agent.model(next_state_tensor)
                        next_state_value = next_value.item()

                    loss = self.agent.compute_loss(log_probs, values, rewards, masks, entropies, next_state_value)
                    self.agent.update_model(loss)
            
                    log_probs = []
                    values = []
                    rewards = []
                    masks = []
                    entropies = []

            if episode == 1:
                running_reward = total_rewards
            else:
                running_reward = 0.05 * total_rewards + 0.95 * running_reward

            raw_history.append(total_rewards)
            smoothed_history.append(running_reward)

            if episode % 10 == 0:
                print(f"Episode {episode} \t Raw {total_rewards} \t Smooth {running_reward}" )
        
        utils.save_model(self.agent)
        utils.plot_learning_curve(raw_rewards = raw_history, smoothed_rewards = smoothed_history)
        self.env.close()
    
            


                
