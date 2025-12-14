import gymnasium as gym
import torch
from Params import params

class Testing():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start_testing(self):
        #load model từ file
        try:
            self.agent.model.load_state_dict(torch.load(params.save_path, map_location=params.device))
            print("Đã load model thành công!")
        except:
            print(f"Chưa có file model tại {params.save_path}")
            return
        
        self.agent.model.eval() # Chuyển sang chế độ test

        state, _ = self.env.reset()
        done = False
        total_reward = 0
            
        while not done:
            # Chuyển state sang tensor
            state_tensor = torch.FloatTensor(state).to(params.device)
                
            # Chỉ cần lấy action từ model, không cần tính toán loss
            with torch.no_grad():
                dist, _ = self.agent.model(state_tensor)
                action = torch.argmax(dist.probs).item() 
                
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
                
        print(f"Điểm số = {total_reward}")

            
        self.env.close()