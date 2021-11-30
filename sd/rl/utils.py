import gym
from gym import spaces
import numpy as np
import time
from pyquaternion import Quaternion

class FlattenWrapper(gym.Wrapper):
    r"""wrapper that flattens observation and actions, and sums the rewards if it's multidimensional"""
    def __init__(self, env):
        super(FlattenWrapper, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)
        self.action_space = spaces.flatten_space(env.action_space)

    def step(self, action):
        unflattened_act = spaces.unflatten(self.env.action_space, action)
        observation, reward, done, info = self.env.step(unflattened_act)
        flattened_obs = spaces.flatten(self.env.observation_space, observation)
        return flattened_obs, np.sum(reward), done, info


    def reset(self):
        return spaces.flatten(self.env.observation_space, super().reset())


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
    Current simulation iteration.
    start_time : timestamp
    Timestamp of the simulation start.
    timestep : float
    Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)


def intrinsic_euler_from_quats(q1: Quaternion, q2: Quaternion, on_error=None):
    try:
        diff: Quaternion = q1.inverse * q2
    except:
        if on_error:
            on_error()
        diff = Quaternion(0.0,0.0,0.0, 1.0)
    # derivative of quaternions as intrinsic euler angles
    intrinsic_euler = q2.rotate(diff.get_axis())*diff.angle
        
    # Represent angular rate as roll pitch yaw, pyquaternion uses yaw roll pitch
    yaw = intrinsic_euler[0]
    roll = intrinsic_euler[1]
    pitch = intrinsic_euler[2]
    return np.array([roll, pitch, yaw])