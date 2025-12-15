import gymnasium as gym

from Params import Params
class Training:
    def __init__(self,env,Agent):
        self.env = env
        self.Agent = Agent

    def run_env(self):
        params = Params() 
        episode_reward = []
        n_episode = params.training_num_episodes
        for episode in n_episode:
            done = False
            state, info = self.env.reset()
            total_rewards = 0
            while not done:
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = truncated or terminated
                total_rewards = total_rewards + reward
                next_value = self.agent.get_value(next_state)
                td_target = reward + params.GAMMA * next_value
                advantage = td_target - value
                self.agent.update(state, action, log_prob(action), advantage, log_prob)
                state = next_state
            episode_reward.append(total_rewards)
        self.env.close()
            


                
