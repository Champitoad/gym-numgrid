import logging
logger = logging.getLogger(__name__)
import numpy as np

import gym
from gym_numgrid import spaces

class Direction(gym.ActionWrapper):
    """
    An action wrapper for NumGrid converting directions into positions.

    Removes position space information at wrapper level, hence should be used after
    any wrapper using it in the wrapper stack; these include:
        - gym_numgrid.wrappers.DiscretePosition
    """
    def __init__(self, env, distance=1):
        super().__init__(env)
        self.distance = distance

        self.digit_space = self.env.digit_space

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
        new_pos = self.env.cursor_pos.copy()
        if not self.direction_space.contains(direction):
            logger.warning("Invalid direction: returning unmoved cursor position")
        else:
            new_pos += np.array(direction) * distance
        return new_pos
