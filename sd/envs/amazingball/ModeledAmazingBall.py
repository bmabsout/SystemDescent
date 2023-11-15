"""
Modeled means the dynamics of the system is represented by a tensorflow operation
Specifically by BallKerasModel
"""

from sd.envs.amazingball.BallKerasModel import amazingball_diff_model
from sd.envs.amazingball.data_env import AmazingBallEnv
from sd import utils
import numpy as np
import gymnasium as gym
import tensorflow as tf
from sd.envs.amazingball.abs_utils import dictionarize_single_system_state, flatten_system_state, dictionarize_system_state
from sd.envs.amazingball.constant import constants

class ModeledAmazingBall(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = AmazingBallEnv(*args, **kwargs)
        self.dynamic_model = amazingball_diff_model() #utils.load_checkpoint(utils.latest_model())
        # gym.make compitability resolution
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.setpoint_space = self.env.setpoint_space

    def run_nn(self, obs, action):
        latent_shape = self.dynamic_model.input["latent"].shape
        latent = np.array([]) if latent_shape[1] == 0 else np.random.normal(latent_shape[1:])
        # batched_obs = utils.map_dict_elems(lambda x: np.array([x]), obs)
        inputs = {'state': np.array([obs]), 'action': np.array([action]), 'latent': np.array([latent])}
        return self.dynamic_model(inputs, training=False)[0]


    def step(self, action):
        # obs = self.state
        # obs = self.env.state
        # obs, rw, done, truncated, i = self.env.step(action)

        flattened_state = flatten_system_state(self.env.state)
        new_flat_state = self.run_nn(flattened_state, action)
        self.env.state = dictionarize_single_system_state(new_flat_state)
        self.env.render()
        return new_flat_state, 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        obs, i = self.env.reset(seed=seed, options=options)
        return flatten_system_state(obs), i
        
if __name__=='__main__':
    env = ModeledAmazingBall(render_mode="human")
    env.reset()
    i = 0

    # draw a random setpoint
    spx, spy = None, None
    ct = 0
    while(1):
        if ct % 300 == 0:
            spx = np.random.uniform(-constants["max_rot_x"], constants["max_rot_x"])
            spy = np.random.uniform(-constants["max_rot_y"], constants["max_rot_y"])
            # spy = np.random.uniform(-env.plate_max_rotation, env.plate_max_rotation)
        # full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
        full_obs, reward, done, truncated, info = env.step(np.array([spx, spy]))
        i+=1
        ct += 1




