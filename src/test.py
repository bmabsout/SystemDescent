import envs.ModeledPendulumDir.ModeledPendulum
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import utils 
try:
	checkpoint_path = utils.latest_model()
	print("using checkpoint:", checkpoint_path)
except e:
	print("there are no trained models")
	exit()


env_name = utils.extract_env_name(checkpoint_path)

env = gym.make('Modeled' + env_name,
	model_path=checkpoint_path)


dynamics = keras.models.load_model(checkpoint_path)


try:
	saved_actor = keras.models.load_model(checkpoint_path + "/actor_tf")
except:
	print(f"there is no actor trained for the model {checkpoint_path}")
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
set_point_angle = -0.
set_point = np.array([np.cos(set_point_angle),np.sin(set_point_angle),0.0])
inputs = np.array([np.cos(thetav), np.sin(thetav), theta_dotv]).T.reshape(-1,3)
set_points = inputs*0 + set_point
print(inputs.shape)
print(set_points.shape)
z = lyapunov([inputs, set_points], training=False)
acts = saved_actor([inputs, set_points], training=False)
print(saved_actor([inputs, set_points]).shape)
after = dynamics([inputs, acts], training=False)
next_z = lyapunov([after, set_points], training=False)
after = utils.to_numpy(after).reshape(pts,pts,3)
z = utils.to_numpy(z).reshape(pts,pts, 1)
next_z = utils.to_numpy(next_z).reshape(pts,pts,1)
acts = utils.to_numpy(acts).reshape(pts,pts,1)
# actor = lambda x, **kwargs: np.array([0.0])
actor = saved_actor
# res = dynamics([states,acts], training=False)

# z = lyapunov([states, set_points], training=False)

# plt.plot(x[:,1], lyapunov([res, set_points]) - y)
plt.pcolormesh(thetav, theta_dotv, z.T[0][:-1, :-1], vmin=0.0, vmax=1.0)
plt.colorbar()
plt.savefig(checkpoint_path + "/lyapunov_tf/lyapunov.png")

# plt.pcolormesh(thetav, theta_dotv, acts.T[0][:-1, :-1])
# plt.colorbar()
# plt.show()


# plt.pcolormesh(thetav, theta_dotv, (next_z - z).T[0][:-1,:-1])
# plt.colorbar()
# plt.show()

orig_env = gym.make(env_name)
seed = np.random.randint(1000000)
# seed = 632732 #bottom almost
# seed = 154911 # almost rotate
# seed = 47039 # almost rotate, then rotate
# seed = 364366 # rotate
print("seed:", seed)
env.seed(seed)
orig_env.seed(seed)
env_obs = env.reset()
orig_env_obs = orig_env.reset()
print("saved_actor")
def feed_obs(obs):
	return [np.array([obs]), np.array([set_point])]

for i in range(200):
	# random_act = np.random.uniform(2,size=(1,))
	act = actor(feed_obs(env_obs), training=False)
	print(env_obs)
	print("lyapunov", lyapunov(feed_obs(env_obs)))
	orig_act = actor(feed_obs(orig_env_obs), training=False)
	env_obs, env_reward, env_done, env_info = env.step(act)
	orig_env_obs, orig_env_reward, orig_env_done, orig_env_info = orig_env.step(orig_act)
	print("error", np.linalg.norm(env_obs-orig_env_obs))
	env.render()
	orig_env.render()
