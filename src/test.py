import envs.KerasPendulumDir.KerasPendulum
import gym
import numpy as np

env = gym.make("Pendulum-v0")
# env = gym.make('KerasPendulum-v0'),
#	model_path="/home/bmabsout/Documents/gymfc-nf1/training_code/neuroflight_trainer/dynamics_learning/saved/ff44f9/checkpoints/checkpoint50.tf")


env.reset()
for i in range(200):
	env.step(np.array([0]))
	env.render()
