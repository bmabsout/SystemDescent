"""
Adapted from 
"""


import time
from typing import TypedDict

import gymnasium as gym
import numpy as np
import pybullet as p
import tensorflow as tf
from gymnasium import spaces
from pybullet_utils import bullet_client as bc
from sd import dfl
from sd.rl import utils
from sd.envs.modelable_env import ModelableEnv, ModelableWrapper
# import Quaternion

def SetpointedAmazingBallEnv(**kwargs):
    return utils.FlattenWrapper(SetpointWrapper(AmazingBallEnv(**kwargs)))



class ObsSpaces(TypedDict):
    velocity: spaces.Box
    position: spaces.Box


class Obs(TypedDict):
    velocity: np.ndarray
    position: np.ndarray


class AmazingBallEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 freq: int=50,
                 real_time: bool=False,
                 aggregate_phy_steps: int=1,
                 user_debug_gui=True,
                 test=False,
                 render_mode="human"
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `AttitudeControlEnv.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        """
        self.test = test
        #### Constants #############################################
        self.M = 0.1
        self.G = 10
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Options ###############################################
        self.GUI = render_mode == "human"
        self.USER_DEBUG = user_debug_gui
        #### Compute constants #####################################
        self.real_time = real_time
        self.START_TIME = time.time()
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = bc.BulletClient(connection_mode=p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                self.CLIENT.configureDebugVisualizer(i, 0)
            self.CLIENT.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0]
                                         )
            ret = self.CLIENT.getDebugVisualizerCamera()
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = np.ones(2)
                self.SLIDERS[0] = self.CLIENT.addUserDebugParameter("x", -0.1, 0.1)
                self.SLIDERS[1] = self.CLIENT.addUserDebugParameter("y", -0.1, 0.1)
                # self.INPUT_SWITCH = self.CLIENT.addUserDebugParameter("Use GUI RPM", 9999, -1, 0)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = bc.BulletClient(connection_mode=p.DIRECT)


        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._obs_space()
        self.init_observation_space = self._init_obs_space()

        #### Housekeeping ##########################################
        self.first_render_call = True
        self._housekeeping()
        self._initialize_pybullet()
        self.CLIENT.stepSimulation()
        self._updateAndStoreKinematicInformation()

    ################################################################################

    def seed(self, seed: int):
        np.random.seed(seed)

    ################################################################################

    def reset_ball(self):
        initial = self.init_observation_space.sample()
        position = np.concatenate((initial["position"], [0.5]))
        velocity = np.concatenate((initial["velocity"], [0.0]))
        self.CLIENT.resetBasePositionAndOrientation(self.ball_id, position, p.getQuaternionFromEuler([0, 0, 0]))
        self.CLIENT.resetBaseVelocity(self.ball_id, velocity)
    def reset_plate(self):
        self.CLIENT.resetBasePositionAndOrientation(self.plate_id, np.array([0,0,0]), p.getQuaternionFromEuler([0, 0, 0]))
        self.CLIENT.resetBaseVelocity(self.plate_id, np.array([0,0,0]))

    def reset(self, seed=None, options=None):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation
        """
        self.seed(seed)
        # self.CLIENT.resetSimulation()
        #### Housekeeping ##########################################
        self.START_TIME = time.time()
        self._housekeeping()
        self.reset_ball()
        self.reset_plate()
        #### Update and store the drones kinematic information #####
        self.CLIENT.stepSimulation()
        self._updateAndStoreKinematicInformation()
        #### Return the initial observation ########################
        return self.obs, {}
    
    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Read the GUI's input parameters #######################
        # if self.USE_GUI_RPM:
        #     for i in range(len(self.SLIDERS)):
        #         self.gui_input[i] = self.CLIENT.readUserDebugParameter(int(self.SLIDERS[i]))
        #     mass = np.tile(self.gui_input, (self.NUM_DRONES, 1))/self.MAX_RPM
        #     if self.step_counter%(self.SIM_FREQ/2) == 0:
        #         self.GUI_INPUT_TEXT = [self.CLIENT.addUserDebugText("Using GUI RPM",
        #                                                   textPosition=[0, 0, 0],
        #                                                   textColorRGB=[1, 0, 0],
        #                                                   lifeTime=1,
        #                                                   textSize=2,
        #                                                   parentObjectUniqueId=self.DRONE_IDS[i],
        #                                                   parentLinkIndex=-1,
        #                                                   replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i])
        #                                                   ) for i in range(self.NUM_DRONES)]
        ### store rpm and action
        self.action = action
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            self._physics(action)
            self.CLIENT.stepSimulation()   ## TODO 
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        ## TODO: set new position based on diffeq : self.CLIENT.resetBasePositionAndOrientation()
        #### Prepare the return values #############################
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        self.last_action = action
        if(self.test):
            utils.sync(self.step_counter, self.START_TIME, self.TIMESTEP)
        return self.obs, 0, False, False, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.simulation_time(), self.SIM_FREQ, (self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))
        print(f"[INFO] BaseAviary.render() ——— plate",
                f"——— velocity {np.array2string(self.obs['velocity'], precision=2)}",
                f" ———  position {np.array2string(self.obs['position'], precision=2)}")
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        self.CLIENT.disconnect()
    
    ################################################################################

    def _initialize_pybullet(self):
        #### Set PyBullet's parameters #############################
        self.CLIENT.setGravity(0, 0, -self.G)
        self.CLIENT.setRealTimeSimulation(self.real_time)
        self.CLIENT.setTimeStep(self.TIMESTEP)
        self.CLIENT.setAdditionalSearchPath("sd/envs/amazingball/assets")
        #### Load plate and ball #########
        self.plate_id = self.CLIENT.loadURDF("plate.urdf")
        self.ball_id = self.CLIENT.createMultiBody(0.2
            , p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
            , basePosition = [0.2,0,0.5]
        )


    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.last_action = np.zeros(2)
        self.gui_input = np.zeros(2)
        self.action = np.zeros(2)
        self.obs = Obs(**self.init_observation_space.sample())


    def _set_obs(self, obs):
        self.obs["position"] = np.clip(obs["position"], self.observation_space["position"].low, self.observation_space["position"].high)
        self.obs["velocity"] = np.clip(obs["velocity"], self.observation_space["velocity"].low, self.observation_space["velocity"].high)

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        pos, quat = self.CLIENT.getBasePositionAndOrientation(self.ball_id)
        vel, _ = self.CLIENT.getBaseVelocity(self.ball_id)
        self._set_obs({"position": pos[0:2], "velocity": vel[0:2]})

    ################################################################################
    def _physics(self, action):
        """Plate angle control via action
        """        
        self.CLIENT.setJointMotorControl2(self.plate_id, 1, p.POSITION_CONTROL, targetPosition=action[0], force=5, maxVelocity=2)
        self.CLIENT.setJointMotorControl2(self.plate_id, 0, p.POSITION_CONTROL, targetPosition=action[1], force=5, maxVelocity=2)

    ################################################################################

        # obs1 = tf.cast(obs1, tf.float32)
    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(2),
            high=np.ones(2)
        )
    
    ################################################################################

    def _init_obs_space(self):
        shape = 2
        return spaces.Dict(ObsSpaces(
            position = spaces.Box(
                low=np.tile(-0.1, shape),
                high=np.tile(0.1, shape),
            ),
            velocity = spaces.Box(
                low=np.tile(0.0, shape),
                high=np.tile(0.0, shape),
            )
        ))


    def _obs_space(self):
        shape = 2
        return spaces.Dict(ObsSpaces(
            position = spaces.Box(
                low=np.tile(-2.0, shape),
                high=np.tile(2.0, shape),
            ),
            velocity = spaces.Box(
                low=np.tile(-0.5, shape),
                high=np.tile(0.5, shape),
            )
        ))


    ################################################################################

    def simulation_time(self):
        return self.step_counter*self.TIMESTEP

    ################################################################################

    def _computeInfo(self):
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

        # ang_v = np.array([[signals.step((-100, 100))(self.simulation_time(), seed) for seed in [121,122,123]]])
        return ({"position": np.array([0.0, 0.0]) , "velocity": np.array([0.0, 0.0])})


if __name__ == "__main__":
    env = SetpointedAmazingBallEnv(render_mode="human", test=True)
    i = 0
    rw_sum = 0
    while(1):
        full_obs, reward, done, truncated, info = env.step(env.action_space.sample())
        rw_sum += reward
        if(i % 100 == 0):
            env.render()
        if(i % 500 == 0):
            print(rw_sum)
            rw_sum = 0
            env.reset()
        i+=1


# TODO: construct a new environmet AmazingBallDataEnv, 
#       where the cooridinate is centered at the plate for math simplicity
#       learn how to write gymnasium env 