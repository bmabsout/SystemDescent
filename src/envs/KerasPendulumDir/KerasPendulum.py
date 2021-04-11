import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import losses


from gym.spaces import Box, Discrete

from gym.envs.registration import register

def random_policy(obs, action_space):
    return action_space.sample()

register(
    id='KerasPendulum-v0',
    entry_point='envs.KerasPendulumDir.KerasPendulum:KerasPendulumEnv',
    max_episode_steps=200,
)

class KerasPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, model_path):

        self.max_speed = 8
        self.max_torque = 2.
        self.viewer = None
        self.step_count = 0

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.seed()
        self.model = keras.models.load_model(model_path)

        def run_nn(obs, action):
            return self.model([np.array([obs]), np.array([action])], training=False)[0]

        print("predicted:", run_nn(np.array([0,1,2]), np.array(0)))
        self.nn = run_nn

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.obs_to_state(self.obs)  # th := theta
        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        g = 10.0
        m = 1
        l = 1
        dt = .05
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        nn_res = self.nn(self.obs, u)
        self.obs = np.array([np.cos(newth), np.sin(newth), newthdot])
        rew = np.linalg.norm(nn_res - self.obs)
        # print(rew)
        # print(nn_res)
        # print(self.obs)
        # if self.step_count % 2 == 0:
        self.obs = nn_res
        self.step_count+=1
        # self.obs[2] = np.clip(self.obs[2], -self.max_speed, self.max_speed)
        return self.obs, rew/200, False, {}

    def obs_to_state(self, obs):
        return [math.atan2(obs[1], obs[0]), obs[2]]

    def reset(self):
        high = np.array([np.pi, 1])
        theta, thetadot = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.obs = np.array([np.cos(theta), np.sin(theta), thetadot])
        self.step_count = 0
        return self.obs

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.0, .4, .7)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(math.atan2(self.obs[1], self.obs[0]) + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
