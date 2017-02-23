from examples.agents.agent import Agent

class RandomAgent(Agent):
    """
    The world's simplest agent!
    """
    def __init__(self, action_space):
        super().__init__(action_space, None)

    def act(self, observation, reward, done, info):
        return self.action_space.sample()
