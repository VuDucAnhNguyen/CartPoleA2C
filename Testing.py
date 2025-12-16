import gymnasium as gym
import torch
from agent import A2CAgent
from params import params

env = gym.make(params.env_name, render_mode="human")
env.reset(seed=params.seed)
torch.manual_seed(params.seed)

agent = A2CAgent(params.input_dim, params.output_dim)
agent.load(params.save_path, eval_mode=True)

for episode in range(5):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _, _, _ = agent.select_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Test Episode {episode+1} | Total Reward: {total_reward}")

env.close()
