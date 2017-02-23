import gym

from gym_numgrid.envs import NumGrid

class NumGridWrapper(gym.Wrapper):
    """
    Base class for NumGrid wrappers conserving additional spaces.
    """
    def __init__(self, env):
        if not (isinstance(env, NumGrid) or isinstance(env, NumGridWrapper)):
            raise TypeError("NumGridWrapper works only with NumGrid and NumGridWrapper subclasses")
        super().__init__(env)

        self.digit_space = self.env.digit_space
        self.position_space = self.env.position_space
