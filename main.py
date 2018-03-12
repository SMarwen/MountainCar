import gym
import numpy as np

from Agent import Agent
from Experiment import Experiment


def main():
    interactive = True
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    # agent = Agent(env.action_space)
    # experiment = Experiment(env, agent)
    # experiment.run(max_number_of_episodes=100, interactive=interactive)



if __name__ == '__main__':
    main()