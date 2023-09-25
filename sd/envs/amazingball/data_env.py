import time
from typing import TypedDict
import gymnasium as gym
import numpy as np
# import pybullet as p
# import tensorflow as tf
from gymnasium import spaces
# from pybullet_utils import bullet_client as bc
from sd import dfl
from sd.rl import utils
from sd.envs.modelable_env import ModelableEnv, ModelableWrapper

def SetpointedAmazingBallEnv(**kwargs):
    return utils.FlattenWrapper(SetpointWrapper(AmazingBallEnv(**kwargs)))

class ObsSpaces(TypedDict):
    velocity: spaces.Box
    position: spaces.Box

class Obs(TypedDict):
    velocity: np.ndarray
    position: np.ndarray

class AmazingBallEnv(gym.Env):

    metadata = {'render.modes': ['human', 'human_describe']}

    def __init__(self,
                 freq: int=50,
                 render_mode="human",
                 plate_angular_v = np.pi / 6,  # assuming 30 degree / sec
                 plate_max_rotation = np.pi / 6, # the max rotation along both axis is 30 deg
                 ball_max_velocity = 10,
                 ball_max_position = 4,
                 dt = 0.01
                ):
        #### Constants #############################################
        self.M = 0.1
        self.G = 10
        self.plate_angular_v = plate_angular_v
        self.plate_max_rotation = plate_max_rotation
        self.ball_max_velocity = ball_max_velocity
        self.ball_max_position = ball_max_position
        self.dt = dt
        #### Compute constants #####################################

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._obs_space()
        self.init_observation_space = self._init_obs_space()

        #### Maintain current observvation #########################
        self.obs = self.init_observation_space.sample()
        self.act = None

        #### Housekeeping ##########################################
        self.render_mode = render_mode
        self.first_render_call = True

    def seed(self, seed: int):
        np.random.seed(seed)

    def _render_human(self):
        raise NotImplementedError
    
    def _render_human_describe(self):
        print(f"[INFO] render ---",
                f"act {self.act}",
                f"———  plat.rot {np.array2string(self.obs['rotation'], precision=5)}",
                f"———  velocity {np.array2string(self.obs['velocity'], precision=5)}",
                f"———  position {np.array2string(self.obs['position'], precision=5)}")

    def render(self):
        if self.render_mode is "human":
            self._render_human()
        elif self.render_mode is "human_describe":
            self._render_human_describe()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_obs = self.init_observation_space.sample()
        self.obs = Obs(**init_obs)
        self.render()
        return self.init_observation_space.sample(), self._get_info()


    def _step_plate(self, action):
        """
        Based on action, step on the rotation of the plate only
        """
        def act2rot(act_n):
            if act_n == 0:
                return 0
            elif act_n == 1:
                return self.plate_angular_v * self.dt
            elif act_n == 2:
                return -self.plate_angular_v * self.dt
            
        plate_rot_x = self.obs["rotation"][0]
        plate_rot_y = self.obs["rotation"][1]
        self.obs["rotation"][0] = np.clip(plate_rot_x + act2rot(action[0]), -self.plate_max_rotation, self.plate_max_rotation)
        self.obs["rotation"][1] = np.clip(plate_rot_y + act2rot(action[1]), -self.plate_max_rotation, self.plate_max_rotation)

    def _step_ball(self):
        """
        Based on the plate rotation, step on the ball
        """
        plate_rot_x = self.obs["rotation"][0]
        plate_rot_y = self.obs["rotation"][1]
        ball_pos_x = self.obs["position"][0]
        ball_pos_y = self.obs["position"][1]
        ball_vel_x = self.obs["velocity"][0]
        ball_vel_y = self.obs["velocity"][1]

        ball_vel_x = ball_vel_x + self.G * np.sin(plate_rot_x) * self.dt
        ball_vel_y = ball_vel_y + self.G * np.sin(plate_rot_y) * self.dt
        ball_pos_x = ball_pos_x + ball_vel_x * self.dt
        ball_pos_y = ball_pos_y + ball_vel_y * self.dt

        self.obs["velocity"][0] = np.clip(ball_vel_x, -self.ball_max_velocity, self.ball_max_velocity)
        self.obs["velocity"][1] = np.clip(ball_vel_y, -self.ball_max_velocity, self.ball_max_velocity)
        self.obs["position"][0] = np.clip(ball_pos_x, -self.ball_max_position, self.ball_max_position)
        self.obs["position"][1] = np.clip(ball_pos_y, -self.ball_max_position, self.ball_max_position)

    def step(self, action):
        self.act = action
        self._step_plate(action)
        self._step_ball() 
        info = self._get_info()
        self.render()
        return self.obs, 0, False, False, info
    

        # self.obs = Obs(**self.init_observation_space.sample())

    def _actionSpace(self):
        """
        action[0] is the x-axis rotation of the plate. value 0: no rotate; 1:positive rotate; 2:negative rotate
        action[1] ...... y-axis ....    
        """
        return spaces.MultiDiscrete([3,3])

    def _init_obs_space(self):
        shape = 2
        return spaces.Dict(ObsSpaces(
            position = spaces.Box(
                low=np.tile(-0.5, shape),
                high=np.tile(0.5, shape),
            ),
            velocity = spaces.Box(
                low=np.tile(-0.5, shape),
                high=np.tile(0.5, shape),
            ),
            rotation = spaces.Box(
                low=np.tile(-self.plate_max_rotation/3, shape),
                high=np.tile(self.plate_max_rotation/3, shape)
            )
        ))

    def _obs_space(self):
        """
        Consideration, the max ball speed can be inferred from max angle. Thus the true ball speed can be calculated. 
        Since action is now limited, the plate theta should also be part of the obs_space"""
        shape = 2
        return spaces.Dict(ObsSpaces(
            position = spaces.Box(
                low=np.tile(-self.ball_max_position, shape),
                high=np.tile(self.ball_max_position, shape),
            ),
            velocity = spaces.Box(
                low=np.tile(-self.ball_max_velocity, shape),
                high=np.tile(self.ball_max_velocity, shape),
            ), 
            rotation = spaces.Box(
                low=np.tile(-self.plate_max_rotation, shape),
                high=np.tile(self.plate_max_rotation, shape)
            )
        ))

    def _get_info(self):
        return {}


class SetpointWrapper(ModelableEnv, gym.Wrapper):
    '''
    Adds the setpoint and rewards so the system forms an MDP to be used with RL algs
    '''
    def __init__(self, env) -> None:
        super().__init__(env)
        self.setpoint = self._calculate_setpoint()
        self.observation_space = self._observationSpace()
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.setpoint = self._calculate_setpoint()
        full_obs = self.observation(obs)
        reward = self.reward(action, full_obs)
        return full_obs, reward, self.done(done, reward), truncated, info

    def reset(self, seed=None, options=None):
        self.setpoint = self._calculate_setpoint()
        obs, i = super().reset(seed=seed, options=options)
        return self.observation(obs), i

    
    def reward(self, action, obs) -> float:
        pos_space = self.env.observation_space["position"]
        vel_space = self.env.observation_space["velocity"]
        max_position_diff = pos_space.high - pos_space.low
        max_velocity_diff = vel_space.high - vel_space.low
        normed_position_error = tf.maximum(tf.abs(obs["state"]["position"] - obs["setpoint"]["position"])/max_position_diff, 1.0)
        normed_velocity_error = tf.maximum(tf.abs(obs["state"]["velocity"] - obs["setpoint"]["velocity"])/max_velocity_diff, 1.0)
        position_closeness = dfl.p_mean(1.0 - normed_position_error, 0.5)
        velocity_closeness = dfl.p_mean(1.0 - normed_velocity_error, 0.5)
        action_smallness = dfl.p_mean(1.0-tf.cast(action, tf.float64), 0.5)
        return action_smallness*position_closeness*velocity_closeness

    def done(self, done, reward):
        return done

    def observation(self, obs):
        return {"state": obs, "setpoint": self.setpoint}

    def _observationSpace(self):
        max_position = np.ones(2)
        max_velocity = 0.5*np.ones(2)
        obs_space = spaces.Dict({
            "state": self.env.observation_space,
            "setpoint": spaces.Dict({
                "position": spaces.Box(low=-max_position, high=max_position),
                "velocity": spaces.Box(low=-max_velocity, high=max_velocity),
            })
        })
        return obs_space
    
        ################################################################################

    def _calculate_setpoint(self):
        """ calculates the desired goal at the current time
        """
        return ({"position": np.array([0.0, 0.0]) , "velocity": np.array([0.0, 0.0])})


if __name__ == "__main__":
    # env = SetpointedAmazingBallEnv(render_mode="human", test=True)
    # i = 0
    # rw_sum = 0
    # while(1):
    #     full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
    #     rw_sum += reward
    #     if(i % 100 == 0):
    #         env.render()
    #     if(i % 500 == 0):
    #         print(rw_sum)
    #         rw_sum = 0
    #         env.reset()
    #     i+=1


    ## Alright, seems working
    env = AmazingBallEnv(render_mode="human_describe")
    env.reset()
    i = 0
    while(1):
        input()
        full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
        i+=1


