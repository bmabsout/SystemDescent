import gym
import numpy as np

class SqueezeWrapper(gym.Wrapper):
    r"""wrapper that squeezes observation and actions"""

    def __init__(self, env):
        super(SqueezeWrapper, self).__init__(env)
        self.observation_space = gym.spaces.flatten_space(env.observation_space)
        self.action_space = gym.spaces.flatten_space(env.action_space)


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return np.squeeze(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(np.reshape(action, self.env.action_space.shape))
        return np.squeeze(observation), np.squeeze(reward), done, info