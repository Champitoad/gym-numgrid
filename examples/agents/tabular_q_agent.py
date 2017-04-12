from collections import defaultdict
import numpy as np
import gym
from gym import spaces

from examples.agents.agent import Agent

class TabularQAgent(Agent):
    """
    Agent implementing tabular Q-learning.
    """
    def __init__(self, action_space, observation_space, **userconfig):
        if not isinstance(action_space, spaces.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        if not isinstance(observation_space, spaces.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        super().__init__(action_space, observation_space)
        self.action_n = action_space.n
        self.config = {
            "init_mean": 0.0,      # Initialize Q values with this mean
            "init_std": 0.0,       # Initialize Q values with this standard deviation
            "learning_rate": 0.1,
            "eps": 0.05,           # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}       # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])
        self.obs = None

    def act(self, observation, reward, done, info):
        # epsilon greedy.
        action = np.argmax(self.q[observation]) if np.random.random() > self.config["eps"] else self.action_space.sample()

        if self.obs is None:
            self.obs = observation
            return action

        future = 0.0
        if not done:
            future = np.max(self.q[observation])
        self.q[self.obs][action] -= \
            self.config["learning_rate"] * (self.q[self.obs][action] - reward - self.config["discount"] * future)

        self.obs = observation
        return action
