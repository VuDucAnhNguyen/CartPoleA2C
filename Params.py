import torch

class Params:
    def __init__ (self):
        self.env_name = "CartPole-v1"
        self.seed = 42

        self.input_dim = 4      # State: [pos, vel, angle, ang_vel]
        self.hidden_dim = 128   # Số neuron lớp ẩn
        self.output_dim = 2     # Action: 0 (Left), 1 (Right)
        
        self.learning_rate = 1e-3  # Tốc độ học (thường A2C dùng 1e-3 hoặc 3e-4)
        self.gamma = 0.99          # Discount factor (trọng số tương lai)
        
        self.training_num_episodes = 1000   # Tổng số màn chơi để train
        self.max_steps = 500       # Số bước tối đa mỗi màn (CartPole v1 max là 500)
        
        self.save_path = "./cartpole_a2c_best.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = Params() 