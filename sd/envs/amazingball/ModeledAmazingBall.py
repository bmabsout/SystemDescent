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


class ModeledAmazingBall(AmazingBallEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_model = amazingball_diff_model() #utils.load_checkpoint(utils.latest_model())

    def run_nn(self, obs, action):
        latent_shape = self.dynamic_model.input["latent"].shape
        latent = np.array([]) if latent_shape[1] == 0 else np.random.normal(latent_shape[1:])
        inputs = utils.map_dict_elems(lambda x: np.array([x]), {  **obs,  "action": action,  "latent": latent })
        return utils.map_dict_elems(lambda x: np.array(x[0]), self.dynamic_model(inputs, training=False))


    def step(self, action):
        obs = self.state
        super().step(action)
        self.state = self.run_nn(obs, action)
        return self._gather_obs(), 0.0, False, False, {}
        
if __name__=='__main__':
    env = ModeledAmazingBall(render_mode="human")
    env.reset()
    i = 0

    # draw a random setpoint
    spx, spy = None, None
    ct = 0
    while(1):
        if ct % 30 == 0:
            spx = np.random.uniform(-env.plate_max_rotation, env.plate_max_rotation)
            spy = np.random.uniform(-env.plate_max_rotation, env.plate_max_rotation)
        # full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
        full_obs, reward, done, truncated, info = env.step(np.array([spx, spy]))
        i+=1
        ct += 1




