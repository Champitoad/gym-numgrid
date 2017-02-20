import logging
logger = logging.getLogger(__name__)

import numpy as np

from gym import ActionWrapper

from gym_numgrid.spaces import Direction

class DirectionWrapper(ActionWrapper):
    """
    An action wrapper for NumGrid converting directions into positions.
    """
    def __init__(self, env, distance=1):
        super().__init__(env)
        self.direction_space = Direction()
        self.distance = distance

    def _action(self, action):
        digit, direction = action
        position = self.cursor_move(direction, self.distance)
        return (digit, position)

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
