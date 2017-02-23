import numpy as np

import gym

from gym_numgrid import spaces
from gym_numgrid.wrappers.numgrid import NumGridWrapper

class DirectionWrapper(NumGridWrapper, gym.ActionWrapper):
    """
    An action wrapper for NumGrid converting directions into positions.

    Since it needs access to the cursor position, which is not saved in
    the wrapper stack, it must be used first in the stack.
    """
    def __init__(self, env, distance=1):
        super().__init__(env)
        self.distance = distance

        self.direction_space = spaces.Direction()

        self.action_space = gym.spaces.Tuple((self.digit_space, self.direction_space))

    def _action(self, action):
        digit, direction = action
        pos = self.cursor_move(direction, self.distance)
        return (digit, pos)

    def cursor_move(self, direction, distance):
        """
        Returns the cursor position if it were moved
        in the given direction on the given distance.
        """
        return self.env.cursor_pos + np.array(direction) * distance
