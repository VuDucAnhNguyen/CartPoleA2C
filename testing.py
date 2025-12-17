import torch
from params import params
from utils import utils

class Testing:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_testing(self):
        try:
            utils.load_model(agent=self.agent)
        except:
            print(f"Chưa có file model tại {params.save_path}")
            return

        self.agent.model.eval()
        state, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(params.device)
            with torch.no_grad():
                probs, _ = self.agent.model(state_tensor)
                action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Điểm số = {total_reward}")
        self.env.close()
