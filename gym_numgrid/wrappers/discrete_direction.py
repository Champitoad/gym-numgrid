import gym

from gym_numgrid.wrappers.direction import Direction

class DiscreteDirection(Direction):
    """
    An action wrapper for NumGrid with a Discrete direction space.
    """
    def __init__(self, env):
        super().__init__(env)
        
        n = len(self.direction_space)
        self.direction_mapping = {self.direction_space.values[i]: i for i in range(n)}
        self.direction_space = gym.spaces.Discrete(n)

        self.action_space = gym.spaces.Tuple(self.digit_space, self.direction_space)

    def _action(self, action):
        super()._action(self.direction_mapping[action])
