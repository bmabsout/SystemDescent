import envs.KerasPendulumDir.KerasPendulum
import gym
import numpy as np
import tensorflow as tf
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

pts = 200

theta = np.linspace(-np.pi, np.pi, pts).reshape(-1,1)
theta_dot = np.linspace(-7.0, 7.0,pts).reshape(-1,1)

thetav, theta_dotv = np.meshgrid(theta, theta_dot)

# states = np.hstack([np.cos(theta), np.sin(theta), theta_dot])
# print(states.shape)
# x = np.linspace([0.,-1,-7], [-2.,1,7],1000)


# friction_actor = friction_actor_def()

# x = np.linspace([0,-1,-7], [-2,1,7],1000)
actor = saved_actor
inputs = np.array([np.cos(thetav), np.sin(thetav), theta_dotv]).T
print(inputs.shape)
z =  tf.make_ndarray(tf.make_tensor_proto(lyapunov(inputs, training=False)))
acts =  tf.make_ndarray(tf.make_tensor_proto(actor(inputs, training=False)))
print(actor(inputs).shape)
after = tf.reshape(dynamics([inputs.reshape(-1,3), acts.reshape(-1,1)], training=False),(pts,pts,3))
next_z = tf.make_ndarray(tf.make_tensor_proto(lyapunov(after, training=False)))

# res = dynamics([states,acts], training=False)

# z = lyapunov(states, training=False)

# plt.plot(x[:,1], lyapunov(res) - y)
plt.pcolormesh(thetav, theta_dotv, z.T[0][:-1, :-1], vmin=0.0, vmax=1.0)
plt.colorbar()
plt.show()

plt.pcolormesh(thetav, theta_dotv, acts.T[0][:-1, :-1])
plt.colorbar()
plt.show()


plt.pcolormesh(thetav, theta_dotv, (next_z - z).T[0][:-1,:-1])
plt.colorbar()
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
