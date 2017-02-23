import gym

from gym_numgrid.wrappers import DirectionWrapper

class DiscreteDirectionWrapper(DirectionWrapper):
    """
    An action wrapper for NumGrid with a Discrete direction space.
    """
    def __init__(self, env):
        super().__init__(env)
        
        n = len(self.direction_space.values)
        self.direction_mapping = {i: self.direction_space.values[i] for i in range(n)}
        self.direction_space = gym.spaces.Discrete(n)

        self.action_space = gym.spaces.Tuple((self.digit_space, self.direction_space))

    def _action(self, action):
        digit, discrete_direction = action
        direction = self.direction_mapping[discrete_direction]
        return super()._action((digit, direction))
