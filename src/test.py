import envs.KerasPendulumDir.KerasPendulum
import gym
import numpy as np
from tensorflow import keras

env = gym.make('KerasPendulum-v0',
	model_path="./saved/428b23/checkpoints/checkpoint210.tf")

saved_actor = keras.models.load_model("./saved/428b23/checkpoints/checkpoint210.tf.actor.tf")
saved_actor.summary()

orig_env = gym.make('Pendulum-v0')
seed = np.random.randint(1000000)
# seed = 154911 # almost rotate
# seed = 47039 # almost rotate, then rotate
# seed = 364366 # rotate
print("seed:", seed)
env.seed(seed)
orig_env.seed(seed)
env_obs = env.reset()
orig_env_obs = orig_env.reset()
print("saved_actor")

for i in range(200):
	# random_act = np.random.uniform(2,size=(1,))
	act = saved_actor.predict([np.array([env_obs])])[0]
	print(env_obs)
	orig_act = saved_actor.predict([np.array([orig_env_obs])])[0]
	env_obs, env_reward, env_done, env_info = env.step(act)
	orig_env_obs, orig_env_reward, orig_env_done, orig_env_info = orig_env.step(orig_act)
	print(np.linalg.norm(env_obs-orig_env_obs))
	env.render()
	orig_env.render()
