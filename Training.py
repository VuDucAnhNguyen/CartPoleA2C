import gymnasium as gym
import torch
from agent import A2CAgent
from params import params

env = gym.make(params.env_name)
env.reset(seed=params.seed)
torch.manual_seed(params.seed)

agent = A2CAgent(params.input_dim, params.output_dim)

for episode in range(1, params.training_num_episodes + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropies = []

    while not done:
        action, log_prob, entropy, value = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        masks.append(torch.tensor(1 - done, dtype=torch.float32))
        entropies.append(entropy)

        state = next_state
        total_reward += reward

    agent.update(log_probs, values, rewards, masks, entropies, next_state)

    if episode % 50 == 0:
        print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

    if episode % 200 == 0:
        agent.save(params.save_path)

agent.save(params.save_path)
env.close()
