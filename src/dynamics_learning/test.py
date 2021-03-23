from KerasPendulumDir.KerasPendulum import KerasPendulumEnv
import gym
import numpy as np

env = KerasPendulumEnv()


env.reset()
for i in range(200):
	env.step(np.array([0]))
	env.render()
