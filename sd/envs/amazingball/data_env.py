import time
from typing import TypedDict
import gymnasium as gym
import numpy as np
# import pybullet as p
# import tensorflow as tf
from gymnasium import spaces
import tensorflow as tf
# from pybullet_utils import bullet_client as bc
from sd import dfl
from sd.rl import utils
from sd.envs.modelable_env import ModelableEnv, ModelableWrapper
import pygame
from pygame import gfxdraw
from sd.envs.amazingball.constant import constants, scale

# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(1400, 900))
# display.start()

def FlattenedAmazingBallEnv(**kwargs):
    return utils.FlattenWrapper(AmazingBallEnv(**kwargs))

class ObsSpaces(TypedDict):
    velocity: spaces.Box
    position: spaces.Box

class Obs(TypedDict):
    velocity: np.ndarray
    position: np.ndarray

class State(TypedDict):
    plate_rot: spaces.Box
    plate_vel: spaces.Box   
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

    Convention:
        plate vel is a tuppe, 0 mean still, 1 mean increase (plate_val), 2 mean decrease (-plate_vel)
    """

    metadata = {'render.modes': ['human', 'human_describe']}

    def __init__(self,
                 render_mode="human",
                 plate_angular_v = constants["pl_vel"],  # assuming 30 degree / sec
                 plate_max_rotation = constants["max_rot_x"], # the max rotation along both axis is 30 deg
                 ball_max_velocity = np.inf,
                 ball_max_position = constants["max_ball_pos_x"],
                 dt = constants["dt"],
                 M = constants["m"],
                 G = constants["g"],
                 collision_damping = constants["collision_damping"]
                ):
        #### Constants #############################################
        self.M = M
        self.G = G
        self.plate_angular_v = plate_angular_v
        self.plate_max_rotation = plate_max_rotation
        self.ball_max_velocity = ball_max_velocity
        self.ball_max_position = ball_max_position
        self.dt = dt
        self.collision_damping = collision_damping
        #### Compute constants #####################################

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.state_space =  self._state_space()
        # self.observation_space =  self._state_space()
        self.observation_space =  self._flat_state_space()
        self.setpoint_space = self._flat_setpoint_space()

        ### Maintain all obj state, so observation can be chosen as a subset #####
        self.state = self._init_state_space().sample()
        # self.state = self._state_space().sample()
        self.act = None

        #### Housekeeping ##########################################
        self.render_mode = render_mode


        #### Rendering group ####################################
        self.screen_dim = 1000
        self.screen = None
        self.clock = None
        self.plate_surface = None
        self.ball_surface = None

    def seed(self, seed: int):
        np.random.seed(seed)

    def _gather_obs(self):
        """ Given a fully updated state, gather the observation """
        obs_keys = ['plate_rot', 'plate_vel', 'ball_pos', 'ball_vel']
        return {k: self.state[k] for k in obs_keys}

    def _render_human(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            self.screen.fill((255, 255, 255))
            self.plate_width = self.screen_dim // 2  # width of plate (adjust as needed)
            self.plate_height = self.screen_dim // 2  # height of plate (adjust as needed)
            abs_pos_plate_topleft = (self.screen_dim - self.plate_width) // 2, (self.screen_dim - self.plate_height) // 2
            self.rend_plate = pygame.Rect(
                abs_pos_plate_topleft[0],
                abs_pos_plate_topleft[1],
                self.plate_width,
                self.plate_height
            )
            self.plate_surface = pygame.Surface((self.plate_width, self.plate_height), pygame.SRCALPHA)
            self.plate_surface.fill((255, 0, 0, 128))  # RGBA where A (alpha) is 128 for 50% transparency

            self.ball_radius = self.plate_width // 24  # radius of ball (adjust as needed)
            self.circle_surface = pygame.Surface((self.ball_radius * 2, self.ball_radius * 2), pygame.SRCALPHA)  # SRCALPHA makes it transparent
            self.circle_surface.fill((255, 255, 255, 128))  
        if self.clock is None:
            self.clock = pygame.time.Clock()

        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # User clicked close button
                running = False
            elif event.type == pygame.KEYDOWN:  # User pressed a key
                if event.key == pygame.K_q:  # If 'q' key was pressed
                    running = False
            elif event.type == pygame.MOUSEBUTTONUP:
                self.state["ball_pos"] = (np.array(pygame.mouse.get_pos()) - np.array(self.rend_plate.center))/self.rend_plate.width * self.ball_max_position*2
                self.state["ball_vel"] = np.array([0, 0])
        if running:
            self._render_human_draw()
        else:
            pygame.quit()

    def _render_human_draw(self):   
        # do a little text description
        # print(f'Current setpoint: {self.act}\r', end='')

        ball_pos_x = scale(self.state["ball_pos"][0], (-self.ball_max_position, self.ball_max_position), (self.rend_plate.topleft[0], self.rend_plate.topleft[0] + self.plate_width))
        ball_pos_y = scale(self.state["ball_pos"][1], (-self.ball_max_position, self.ball_max_position), (self.rend_plate.topleft[1], self.rend_plate.topleft[1] + self.plate_height))
        if isinstance(ball_pos_x, tf.Tensor):
            ball_pos_x = ball_pos_x.numpy().item()
            ball_pos_y = ball_pos_y.numpy().item()
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (255, 255, 255), self.rend_plate)
        self.screen.blit(self.plate_surface, self.rend_plate.topleft)
        pygame.draw.circle(self.screen, (0, 0, 255), (ball_pos_x, ball_pos_y), self.ball_radius)
        # pygame.draw.circle(self.screen, (0, 0, 255), (0, 0), self.ball_radius)

        # draw stripe to indicate the rotation of the plate
        margin = constants['render_stripe_margin']
        thickness = constants['render_tiltline_thickness']
        rot_x, rot_y = self.state["plate_rot"]
        if isinstance(rot_x, tf.Tensor):
            rot_x = rot_x.numpy().item()
            rot_y = rot_y.numpy().item()

        offset = scale(rot_x, (-self.plate_max_rotation, self.plate_max_rotation), (-self.plate_width/2, self.plate_width/2))
        if isinstance(offset, tf.Tensor):
            offset = offset.numpy().item()
        pygame.draw.line(self.screen, (0, 255, 0), (self.rend_plate.centerx, self.rend_plate.top - margin), (self.rend_plate.centerx + offset, self.rend_plate.top - margin), thickness)
        pygame.draw.line(self.screen, (0, 255, 0), (self.rend_plate.centerx, self.rend_plate.bottom + margin), (self.rend_plate.centerx + offset, self.rend_plate.bottom + margin), thickness)
        offset = scale(rot_y, (-self.plate_max_rotation, self.plate_max_rotation), (-self.plate_height/2, self.plate_height/2))
        pygame.draw.line(self.screen, (0, 255, 0), (self.rend_plate.left - margin, self.rend_plate.centery), (self.rend_plate.left - margin, self.rend_plate.centery + offset), thickness)
        pygame.draw.line(self.screen, (0, 255, 0), (self.rend_plate.right + margin, self.rend_plate.centery), (self.rend_plate.right + margin, self.rend_plate.centery + offset), thickness)

        # draw big cross spanning the entire screen
        pygame.draw.line(self.screen, (200, 200, 200), (0, self.screen_dim/2), (self.screen_dim, self.screen_dim/2), 5)
        pygame.draw.line(self.screen, (200, 200, 200), (self.screen_dim/2, 0), (self.screen_dim/2, self.screen_dim), 5)

        # text group
        # font = pygame.font.SysFont('Arial', 20)

        pygame.display.flip()
        pygame.time.wait(int(self.dt*1000)) # wait for dt seconds, .wait's arg is in ms
    
    def _render_human_describe(self):
        print(f"[INFO] render ---",
                f"act {self.act}",
                f"———  plat.rot {np.array2string(self.state['plate_rot'], precision=5)}",
                f"———  ball.pos {np.array2string(self.state['ball_pos'], precision=5)}",
                f"———  ball.vel {np.array2string(self.state['ball_vel'], precision=5)}")

    def render(self):
        if self.render_mode is "human":
            self._render_human()
        elif self.render_mode is "human_describe":
            self._render_human_describe()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._state_space().sample()
        # self.state = self._init_state_space().sample()
        # self.render()
        return self._gather_obs(), self._get_info()
        # return self.state, self._get_info()


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
                self.state["plate_vel"][i] = 0* self.plate_angular_v
            else:
                self.state["plate_rot"][i] += np.sign(sp - self.state["plate_rot"][i]) * self.plate_angular_v * self.dt
                # self.state["plate_vel"] = _assign_tuple_element(self.state["plate_vel"], i, 1) if sp > self.state["plate_rot"][i] else _assign_tuple_element(self.state["plate_vel"], i, 2)
                self.state["plate_vel"][i] = self.plate_angular_v if sp > self.state["plate_rot"][i] else -self.plate_angular_v
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
            # collision handle 1: if the ball_pos is out of bound, then set the velocity  to  zero
            # if abs(ball_pos) >= self.ball_max_position:
            #     ball_vel = 0 * ball_vel

            # collision handle 2: if the ball_pos is out of bound, then set the velocity to -vel
            if abs(ball_pos) >= self.ball_max_position:
                ball_vel = -ball_vel * self.collision_damping

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
        # return self.state, 0, False, False, info
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
            # plate_vel = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))),
            plate_vel = spaces.Box(
                low=np.tile(-self.plate_angular_v, shape),
                high=np.tile(self.plate_angular_v, shape)
            ),
            ball_pos = spaces.Box(
                low=np.tile(-self.ball_max_position, shape),
                high=np.tile(self.ball_max_position, shape),
            ),
            ball_vel = spaces.Box(
                low=np.tile(-self.ball_max_velocity, shape),
                high=np.tile(self.ball_max_velocity, shape),
            ),
        ))

    def _flat_state_space(self):
        shape = 2
        # Combine low bounds
        low = np.concatenate([
            np.tile(-self.plate_max_rotation, shape),
            np.tile(-self.plate_angular_v, shape),
            np.tile(-self.ball_max_position, shape),
            np.tile(-self.ball_max_velocity, shape)
        ])

        # Combine high bounds
        high = np.concatenate([
            np.tile(self.plate_max_rotation, shape),
            np.tile(self.plate_angular_v, shape),
            np.tile(self.ball_max_position, shape),
            np.tile(self.ball_max_velocity, shape)
        ])

        # Create a single Box space
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _flat_setpoint_space(self):
        shape = 2
        # Combine low bounds
        low = np.concatenate([
            np.tile(-self.ball_max_position, shape),
            np.tile(-self.ball_max_velocity, shape)
        ])

        # Combine high bounds
        high = np.concatenate([
            np.tile(self.ball_max_position, shape),
            np.tile(self.ball_max_velocity, shape)
        ])

        # Create a single Box space
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _init_state_space(self):
        shape = 2
        return spaces.Dict(State(
            plate_rot = spaces.Box(
                low=np.tile(-self.plate_max_rotation/3, shape),
                high=np.tile(self.plate_max_rotation/3, shape)
            ),
            plate_vel = spaces.Box(
                low=np.tile(0, shape),
                high=np.tile(0, shape)
            ),
            ball_pos = spaces.Box(
                low=np.tile(-self.ball_max_position/1.2, shape),
                high=np.tile(self.ball_max_position/1.2, shape),
            ),
            ball_vel = spaces.Box(
                low=np.tile(-1, shape),
                high=np.tile(1, shape),
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
    env = AmazingBallEnv(render_mode="human")
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


