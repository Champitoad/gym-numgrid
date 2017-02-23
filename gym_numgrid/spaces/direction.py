import gym
from gym import spaces

class Direction(gym.Space):
    """
    Discrete space consisting of the 4 orthogonal directions.
    """
    def __init__(self):
        # In order: left, right, top, bottom
        self.values = [(-1,0), (1,0), (0,-1), (0,1)]

    def sample(self):
        spaces.prng.np_random.shuffle(self.values)
        return self.values[0]

    def contains(self, x):
        return x in self.values
