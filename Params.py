import torch

class Params:
    def __init__(self):
        # Môi trường
        self.env_name = "CartPole-v1"
        self.seed = 42

        # Kích thước mạng
        self.input_dim = 4
        self.hidden_dim = 128
        self.output_dim = 2

        # Hyperparameters
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.n_steps = 5           # n-step cho A2C
        self.entropy_beta = 0.01   # entropy coefficient

        # Training
        self.training_num_episodes = 1000
        self.max_steps = 500       # CartPole-v1 max 500

        # Save/load
        self.save_path = "./cartpole_a2c_best.pth"

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = Params()
