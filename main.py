import gymnasium as gym
from params import params
from agent import A2CAgent
from training import Training
from testing import Testing

class Main:
    def __init__(self, mode):
        self.mode = mode
        self.agent = A2CAgent(input_dim = params.input_dim, n_actions = params.output_dim)

        if (mode == 0):
            self.env = gym.make(params.env_name, render_mode = None)
            self.trainer = Training(agent=self.agent, env=self.env)
        else:
            self.env = gym.make(params.env_name, render_mode = "rgb_array", max_episode_steps= params.max_steps)
            self.tester = Testing(agent=self.agent, env=self.env)

    def run(self):
        if (self.mode == 0):
            self.trainer.start_training()
        else:
            self.tester.start_testing()


if (__name__ == "__main__"):
    try:
        user_input = input("Mode: Training(0), Test(1): ")
        mode = int(user_input)
        if (mode not in [0, 1]):
            print("Invalid input, default mode: Training")
            mode = 0
    except:
        print("Invalid input, default mode: Training")
        mode = 0

    A2C = Main(mode = mode)
    A2C.run()