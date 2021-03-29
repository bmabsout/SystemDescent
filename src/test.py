import envs.KerasPendulumDir.KerasPendulum
import gym
import numpy as np

env = gym.make('KerasPendulum-v0',
	model_path="./saved/256769/checkpoints/checkpoint40.tf")


orig_env = gym.make('Pendulum-v0')
# seed = np.random.randint(1000000)
# seed = 154911 # almost rotate
seed = 47039 # almost rotate, then rotate
# seed = 364366 # rotate
print("seed:", seed)
env.seed(seed)
orig_env.seed(seed)
env.reset()
orig_env.reset()

for i in range(200):
	random_act = np.random.uniform(2,size=(1,))
	env_obs, env_reward, env_done, env_info = env.step(random_act)
	orig_env_obs, orig_env_reward, orig_env_done, orig_env_info = orig_env.step(random_act)
	print(np.linalg.norm(env_obs-orig_env_obs))
	env.render()
	orig_env.render()
