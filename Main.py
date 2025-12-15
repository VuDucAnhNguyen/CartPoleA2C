import gymnasium as gym
from Params import params
from Agent import Agent
from Training import Training
from Testing import Testing

class Main:
    def __init__(self, mode):
        self.mode = mode
        self.agent = Agent(params.input_dim, params.output_dim)

        if (mode == 0):
            self.env = gym.make(params.env_name, render_mode = None)
            self.trainer = Training(agent=self.agent, env=self.env)
        else:
            self.env = gym.make(params.env_name, render_mode = "human")
            self.tester = Testing(agent=self.agent, env=self.env)


    def run(self):
        if (self.mode == 0):
            self.trainer.run_env()
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