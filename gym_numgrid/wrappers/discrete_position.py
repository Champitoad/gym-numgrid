import gym

from gym_numgrid.utils.spaces import total_discrete_mapping
from gym_numgrid.wrappers.numgrid import NumGridWrapper

class DiscretePositionWrapper(NumGridWrapper, gym.ActionWrapper):
    """
    An action wrapper for NumGrid with a Discrete position space.
    """
    def __init__(self, env):
        super().__init__(env)

        pos_mapping = total_discrete_mapping(self.position_space)
        self.position_space = gym.spaces.DiscreteToMultiDiscrete(self.position_space, pos_mapping)

        self.action_space = gym.spaces.Tuple((self.digit_space, self.position_space))

    def _action(self, action):
        digit, pos = action
        pos = self.position_space(pos)
        return (digit, pos)
