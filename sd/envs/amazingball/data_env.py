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

class State(TypedDict):
    plate_rot: spaces.Box
    plate_vel: spaces.Tuple   # convention: 0 still, 1 increase, 2 decrease
    ball_pos:  spaces.Box
    ball_vel:  spaces.Box


class AmazingBallEnv(gym.Env):
    """
    Whether make the rotation of the plate as observation
    Whether make the ang velo of the plate as observation
        - the action is chosen to be the setpoint of rotation
        - If the servo is using PID to tilt the plate, then the rotation and angular velocity is visible
        - Consider the DNN would control the entire sys, then the two param should be part of the observation

    Design philosophy:
        the environment is completely described by the state
        per reset/step, the environment only acts upon state
        From the observation would then be derived from the state
        Under this principle, the visible and hidden observation can be chosen as a subset of the state
    """

    metadata = {'render.modes': ['human', 'human_describe']}

    def __init__(self,
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

        ### Maintain all obj state, so observation can be chosen as a subset #####
        self.state = self._init_state_space().sample()
        self.act = None

        #### Housekeeping ##########################################
        self.render_mode = render_mode

    def seed(self, seed: int):
        np.random.seed(seed)

    def _gather_obs(self):
        """ Given a fully updated state, gather the observation """
        obs_keys = ['plate_rot', 'ball_pos', 'ball_vel']
        return {k: self.state[k] for k in obs_keys}

    def _render_human(self):
        raise NotImplementedError
    
    def _render_human_describe(self):
        print(f"[INFO] render ---",
                f"act {self.act}",
                f"———  plat.rot {np.array2string(self.state['plate_rot'], precision=5)}",
                f"———  ball.pos {np.array2string(self.state['ball_pos'], precision=5)}",
                f"———  ball.vel {np.array2string(self.state['ball_vel'], precision=5)}")
                # f"———  plat.rot {np.array2string(self.obs['rotation'], precision=5)}",
                # f"———  velocity {np.array2string(self.obs['velocity'], precision=5)}",
                # f"———  position {np.array2string(self.obs['position'], precision=5)}")

    def render(self):
        if self.render_mode is "human":
            self._render_human()
        elif self.render_mode is "human_describe":
            self._render_human_describe()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._init_state_space().sample()
        self.render()
        print('reset request')
        print(self._gather_obs())
        return self._gather_obs(), self._get_info()


    def _step_plate(self, action):
        """
        Based on action, step on the rotation of the plate only
        """
        def _assign_tuple_element(t, i, v):
            l = list(t)
            l[i] = v
            return tuple(l)

        def _step_rot_axis(i):
            sp = action[i]
            if abs(self.state["plate_rot"][i] - sp) <= self.plate_angular_v * self.dt:
                self.state["plate_rot"][i] = sp
                _assign_tuple_element(self.state["plate_vel"], i, 0)
            else:
                self.state["plate_rot"][i] += np.sign(sp - self.state["plate_rot"][i]) * self.plate_angular_v * self.dt
                _assign_tuple_element(self.state["plate_vel"], i, 1) if sp > self.state["plate_rot"][i] else _assign_tuple_element(self.state["plate_vel"], i, 2)
            self.state["plate_rot"][i] = np.clip(self.state["plate_rot"][i], -self.plate_max_rotation, self.plate_max_rotation)

        _step_rot_axis(0)
        _step_rot_axis(1)


    def _step_ball(self):
        """
        Based on the plate rotation, step on the ball
        """
        def _step_ball_axis(i):
            ball_pos = self.state["ball_pos"][i]
            ball_vel = self.state["ball_vel"][i]
            plate_rot = self.state["plate_rot"][i]
            ball_vel = ball_vel + self.G * np.sin(plate_rot) * self.dt
            ball_pos = ball_pos + ball_vel * self.dt
            # if the ball_pos is out of bound, then set the velocity  to  zero
            if abs(ball_pos) >= self.ball_max_position:
                ball_vel = 0
            self.state["ball_vel"][i] = np.clip(ball_vel, -self.ball_max_velocity, self.ball_max_velocity)
            self.state["ball_pos"][i] = np.clip(ball_pos, -self.ball_max_position, self.ball_max_position)

        _step_ball_axis(0)
        _step_ball_axis(1)

    def step(self, action):
        self.act = action
        self._step_plate(action)
        self._step_ball() 
        info = self._get_info()
        self.render()
        return self._gather_obs(), 0, False, False, info

    def _actionSpace(self):
        """
        The action is a 2D vector, each is the setpoint (plate rotation) along each axis
        """
        return spaces.Box(
            low=np.array([-self.plate_max_rotation, -self.plate_max_rotation]),
            high=np.array([self.plate_max_rotation, self.plate_max_rotation]),
        )

    def _state_space(self):
        shape = 2
        return spaces.Dict(State(
            plate_rot = spaces.Box(
                low=np.tile(-self.plate_max_rotation, shape),
                high=np.tile(self.plate_max_rotation, shape)
            ),
            plate_vel = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))),
            ball_pos = spaces.Box(
                low=np.tile(-self.ball_max_position, shape),
                high=np.tile(self.ball_max_position, shape),
            ),
            ball_vel = spaces.Box(
                low=np.tile(-self.ball_max_velocity, shape),
                high=np.tile(self.ball_max_velocity, shape),
            ),
        ))

    def _init_state_space(self):
        shape = 2
        return spaces.Dict(State(
            plate_rot = spaces.Box(
                low=np.tile(-self.plate_max_rotation/3, shape),
                high=np.tile(self.plate_max_rotation/3, shape)
            ),
            plate_vel = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))),
            ball_pos = spaces.Box(
                low=np.tile(-self.ball_max_position/3, shape),
                high=np.tile(self.ball_max_position/3, shape),
            ),
            ball_vel = spaces.Box(
                low=np.tile(-self.ball_max_velocity/3, shape),
                high=np.tile(self.ball_max_velocity/3, shape),
            ),
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
        # full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
        full_obs, reward, done, truncated, info = env.step(np.array([0.1, 0.1]))
        i+=1


