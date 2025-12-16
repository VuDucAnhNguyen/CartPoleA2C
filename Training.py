import gymnasium as gym
import torch
from Agent import A2CAgent
from Params import params  # Import Params instance

def train():
    # 1️⃣ Khởi tạo môi trường
    env = gym.make(params.env_name)
    torch.manual_seed(params.seed)

    # 2️⃣ Khởi tạo agent
    agent = A2CAgent(input_dim=params.input_dim, n_actions=params.output_dim)

    for episode in range(1, params.training_num_episodes + 1):
        state, _ = env.reset(seed=params.seed)  # ✅ seed môi trường và unpack info
        log_probs = []
        state_values = []
        rewards = []
        masks = []

        total_reward = 0

        for step in range(params.max_steps):
            action, log_prob, state_value = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)  # ✅ Gymnasium
            done = terminated or truncated

            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)
            masks.append(1 - int(done))

            state = next_state
            total_reward += reward

            if done:
                break

        loss = agent.compute_loss(log_probs, state_values, rewards, masks)
        agent.update_model(loss)

        if episode % 10 == 0:
            print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Loss: {loss.item():.4f}")

    agent.save_model(params.save_path)
    print(f"Training completed. Model saved to {params.save_path}")


if __name__ == "__main__":
    train()
