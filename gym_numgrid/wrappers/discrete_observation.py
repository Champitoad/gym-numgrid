import gym

from gym_numgrid.utils.spaces import total_discrete_mapping
from gym_numgrid.wrappers.numgrid import NumGridWrapper

class DiscreteObservationWrapper(NumGridWrapper, gym.ObservationWrapper):
    """
    An observation wrapper for NumGrid with a Discrete observation space,
    for compatibility with agents such as the tabular Q-learning algorithm.
    """
    def __init__(self, env):
        super().__init__(env)

        pos_mapping = total_discrete_mapping(self.observation_space)
        self.observation_space = gym.spaces.DiscreteToMultiDiscrete(self.observation_space, pos_mapping)

    def _observation(self, observation):
        discrete = list(self.observation_space.mapping.keys())
        multi_discrete = list(self.observation_space.mapping.values())
        return discrete[multi_discrete.index(observation.tolist())]
