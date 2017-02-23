class Agent:
    """
    Abstract class defining an agent properties and interface.
    """
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation, reward, done, info):
        raise NotImplementedError
