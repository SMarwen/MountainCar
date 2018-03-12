from collections import defaultdict
import numpy as np
import tensorflow as tf


class Agent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return np.random.choice(self.action_space)

    def learn(self):
        pass