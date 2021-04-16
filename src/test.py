import envs.KerasPendulumDir.KerasPendulum
import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

checkpoint_path = "./saved/f6bfa5/checkpoints/checkpoint4"

env = gym.make('KerasPendulum-v0',
	model_path=checkpoint_path)

dynamics = keras.models.load_model(checkpoint_path)

saved_actor = keras.models.load_model(checkpoint_path + "/actor_tf")
saved_actor.summary()

lyapunov = keras.models.load_model(checkpoint_path + "/lyapunov_tf")

x = np.linspace([0.,-1,-7], [2.,1,7],1000)


# friction_actor = friction_actor_def()

x = np.linspace([0,-1,-7], [-2,1,7],1000)
actor = saved_actor

acts =  actor(x, training=False)

res = dynamics([x,acts], training=False)

y = lyapunov(x, training=False)

plt.plot(x[:,1], lyapunov(res) - y)
plt.show()


plt.plot(x[:,1],y)
plt.show()

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

	act = actor(np.array([env_obs]), training=False)
	print(env_obs)
	print("lyapunov", lyapunov(np.array([env_obs])))
	orig_act = actor(np.array([orig_env_obs]), training=False)
	env_obs, env_reward, env_done, env_info = env.step(act)
	orig_env_obs, orig_env_reward, orig_env_done, orig_env_info = orig_env.step(orig_act)
	print("error", np.linalg.norm(env_obs-orig_env_obs))
	env.render()
	orig_env.render()
