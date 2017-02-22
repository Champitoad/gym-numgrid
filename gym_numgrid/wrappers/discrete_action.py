import gym

from gym_numgrid.utils.spaces import total_discrete_mapping

class DiscreteAction(gym.ActionWrapper):
    """
    An action wrapper for NumGrid with a Discrete action space,
    for compatibility with agents such as the tabular Q-learning algorithm.
    """
    def __init__(self, env):
        super().__init__(env)

        bounds = [space.n - 1 for space in self.action_space.spaces]
        params = np.stack([np.zeros(len(self.action_space.spaces)), bounds], 1)

        action_space = gym.spaces.MultiDiscrete(params)
        action_mapping = total_discrete_mapping(action_space)

        self.action_space = gym.spaces.DiscreteToMultiDiscrete(action_space, action_mapping)

    def _action(self, action):
        return self.action_space[action]
