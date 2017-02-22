import gym

from gym_numgrid.utils.spaces import total_discrete_mapping

class DiscreteObservation(gym.ObservationWrapper):
    """
    An observation wrapper for NumGrid with a Discrete observation space,
    for compatibility with agents such as the tabular Q-learning algorithm.
    """
    def __init__(self, env):
        super().__init__(env)

        self.digit_space = self.env.digit_space

        pos_mapping = total_discrete_mapping(self.observation_space)
        self.observation_space = gym.spaces.DiscreteToMultiDiscrete(self.observation_space, pos_mapping)

    def _observation(self, observation):
        return self.observation_space[observation]
