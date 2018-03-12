import gym
import numpy as np

from Agent import Agent
from Experiment import Experiment


def main():
    interactive = False
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    action_space_list = list(set([env.action_space.sample() for i in range(100)]))
    agent = Agent(action_space_list)
    experiment = Experiment(env, agent)
    experiment.run(max_number_of_episodes=100, interactive=interactive)


if __name__ == '__main__':
    main()
