"""
Adapted from 
"""


import time
from typing import TypedDict
import xml.etree.ElementTree as etxml

import gym
import numpy as np
import pybullet as p
import pybullet_data
import sd.envs.drone.signals as signals
import tensorflow as tf
from gym import spaces
from pybullet_utils import bullet_client as bc
from sd import dfl
from sd.envs.drone.assets import load_assets
from sd.envs.modelable_env import ModelableEnv, ModelableWrapper
from sd.rl import utils
from dataclasses import dataclass
from pyquaternion import Quaternion
# import Quaternion

def ModelableMultiAttitudeEnv(**kwargs):
    return DistanceWrapper(MultiAttitudeEnv(**kwargs))

def AttitudeEnv(**kwargs):
    return utils.FlattenWrapper(ModelableMultiAttitudeEnv(**kwargs))

def SetpointedAttitudeEnv(**kwargs):
    return utils.FlattenWrapper(SetpointWrapper(ModelableMultiAttitudeEnv(**kwargs)))

class DistanceWrapper(ModelableWrapper):
    @staticmethod
    @tf.function
    def closeness_dfl(obs1: tf.Tensor, obs2: tf.Tensor) -> dfl.DFL:
        # dot = tf.tensordot(obs1, obs2, axes=3)
        # obs1_size = tf.linalg.norm(obs1)
        # obs2_size = tf.linalg.norm(obs2)
        # length_sim = 5 / (5 + tf.abs(obs1_size - obs2_size))
        # if(obs1_size < 0.01 or obs2_size < 0.01):
        #     cos_sim = 1.0
        # else:
        #     cos_sim = (dot/(obs1_size * obs2_size)+1.0)/2.0
        # return cos_sim*length_sim
        # tf.print(obs1)
        max_ang_v = tf.ones((obs1.shape[0], 3)) * 3000.0
        # max_obs = max_ang_v
        # min_obs = -max_ang_v
        max_ang_acc = tf.ones((obs1.shape[0], 3)) * 3000.0
        max_obs = tf.concat([max_ang_v, max_ang_acc], axis=1)
        min_obs = tf.concat([-max_ang_v, -max_ang_acc], axis=1)
        normed_obs1 = (obs1 - min_obs)/(max_obs - min_obs)
        normed_obs2 = (obs2 - min_obs)/(max_obs - min_obs)
        # tf.print(-dfl.p_mean(abs_diff, 1.0))
        # rpy_sims = dfl.geo(10.0/(10.0 + abs_diff), axis=2)
        normed_closeness = 1.0 - tf.abs(normed_obs1 - normed_obs2)
        
        # r = (1)
        # print(self.obs["ang_v"], "\t", r)
        return dfl.p_mean(dfl.smooth_constraint(normed_closeness, 0.5, 1.0), 0.0)



class ObsSpaces(TypedDict):
    ang_v: spaces.Box
    ang_acc: spaces.Box


class Obs(TypedDict):
    ang_v: np.ndarray
    ang_acc: np.ndarray


def rpy_to_pybullet(rpy):
    return np.array([-rpy[1], rpy[0], -rpy[2]])

class MultiAttitudeEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 num_drones: int=1,
                 freq: int=100,
                 real_time: bool=False,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 urdf_path=load_assets.get_urdf_path("cf2x.urdf"),
                 user_debug_gui=True,
                 test=False,
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
        self.G = 0
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        #### Options ###############################################
        self.GUI = gui
        self.USER_DEBUG = user_debug_gui
        self.URDF = urdf_path
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.MAX_RPM, \
        self.KF, \
        self.KM, \
        self.MAX_SPEED_KMH, \
        self.DRAG_COEFF = self._parseURDFParameters()
        print(f"""[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:
                  [INFO] m {self.M}, L {self.L},[INFO] kf {self.KF}, km {self.KM}
                  [INFO] max_rpm {self.MAX_RPM}, max_speed_kmh {self.MAX_SPEED_KMH}
                  [INFO] drag_xy_coeff {self.DRAG_COEFF[0]}, drag_z_coeff {self.DRAG_COEFF[2]}
                  """)
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
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = self.CLIENT.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM)
                self.INPUT_SWITCH = self.CLIENT.addUserDebugParameter("Use GUI RPM", 9999, -1, 0)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = bc.BulletClient(connection_mode=p.DIRECT)

        #### Set initial poses #####################################
        self.INIT_XYZS = np.array([0,0,1]).reshape(1, 3)

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._obs_space()
        self.init_observation_space = self._init_obs_space()

        #### Housekeeping ##########################################
        self.first_render_call = True
        self._housekeeping()
        self._initialize_pybullet()
        self.CLIENT.stepSimulation()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()

    ################################################################################

    def seed(self, seed: int):
        np.random.seed(seed)

    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation
        """
        # self.CLIENT.resetSimulation()
        #### Housekeeping ##########################################
        self.START_TIME = time.time()
        self._housekeeping()
        for droneId, i in zip(self.DRONE_IDS, range(len(self.DRONE_IDS))):
            self.CLIENT.resetBasePositionAndOrientation(droneId, self.INIT_XYZS[i], self.quat[i])
            self.CLIENT.resetBaseVelocity(droneId, self.vel[i], rpy_to_pybullet(self.obs["ang_v"][i]*self.DEG2RAD))
        #### Update and store the drones kinematic information #####
        self.CLIENT.stepSimulation()
        self._updateAndStoreKinematicInformation()
        #### Return the initial observation ########################
        return self.obs
    
    ################################################################################

    def step(self,
             action
             ):
        # print(action)
        action = action/2.0 + 0.5
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
        if self.GUI and self.USER_DEBUG:
            current_input_switch = self.CLIENT.readUserDebugParameter(self.INPUT_SWITCH)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = not self.USE_GUI_RPM
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = self.CLIENT.readUserDebugParameter(int(self.SLIDERS[i]))
            action = np.tile(self.gui_input, (self.NUM_DRONES, 1))/self.MAX_RPM
            if self.step_counter%(self.SIM_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [self.CLIENT.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i])
                                                          ) for i in range(self.NUM_DRONES)]
        ### store rpm and action
        self.action = action
        self.rpm = action * self.MAX_RPM
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            self._physics()
            self.CLIENT.stepSimulation()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        self._saveLastAction(action)
        if(self.test):
            utils.sync(self.step_counter, self.START_TIME, self.TIMESTEP)
        return self.obs, 0, False, info
    
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
        for i in range (self.NUM_DRONES):
            ang_v = (self.obs["ang_v"][i])
            print(f"[INFO] BaseAviary.render() ——— drone {i}",
                  f"——— ang_v {np.array2string(ang_v, precision=2)}",
                  f" ———  rpm {np.array2string(self.rpm[i], precision=2)}")
    
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
        self.CLIENT.setAdditionalSearchPath(pybullet_data.getDataPath())
        #### Load ground plane and drone #########
        self.PLANE_ID = self.CLIENT.loadURDF("plane.urdf")
        self.DRONE_IDS = np.array([self.CLIENT.loadURDF(self.URDF,
                                              self.INIT_XYZS[i,:],
                                              self.quat[i],
                                              flags = p.URDF_USE_INERTIA_FROM_FILE
                                              ) for i in range(self.NUM_DRONES)])
        
        self._creates_spherical_constraints()
        
        self._create_link_dicts()

        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
        self.debug_force_line = -np.ones(4)
        

        for droneId, i in zip(self.DRONE_IDS, range(len(self.DRONE_IDS))):
            self.CLIENT.resetBaseVelocity(droneId, self.vel[i], rpy_to_pybullet(self.obs["ang_v"][i]*self.DEG2RAD))

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        self.action = np.zeros((self.NUM_DRONES, 4))
        self.rpm = np.zeros((self.NUM_DRONES, 4))
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, self.NUM_DRONES))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.obs = Obs(**self.init_observation_space.sample())

    ################################################################################

    def _create_link_dicts(self):
        """ Creates an array containing each drone's dictionary of link names -> ids
        """
        self.drone_to_link_name_to_index = []
        for drone_id in self.DRONE_IDS:
            link_dict = {self.CLIENT.getBodyInfo(drone_id): -1, }
            self.drone_to_link_name_to_index.append(link_dict)
            for link_id in range(p.getNumJoints(drone_id)):
                link_name = p.getJointInfo(drone_id, link_id)[12].decode('UTF-8')
                link_dict[link_name] = link_id

    
    ################################################################################

    def _creates_spherical_constraints(self):
        """ Creates the constraint keeping the drones attached around their center of gravity
        """
        for drone_id, location in zip(self.DRONE_IDS, self.INIT_XYZS):
            joint_id = self.CLIENT.createConstraint(
                drone_id, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], location)
            # self.CLIENT.changeDynamics(drone_id, joint_id, maxJointVelocity=100.0) #100 rad/s

    ################################################################################

    def on_quaternion_error(self):
        print("error")
        self.reset()

    def _update_ang_v(self, ang_v):
        ang_v_space = self.observation_space["ang_v"]
        self.obs["ang_v"] = np.clip(ang_v, ang_v_space.low, ang_v_space.high)

    def _update_ang_acc(self, ang_acc):
        ang_acc_space = self.observation_space["ang_acc"]
        self.obs["ang_acc"] = np.clip(ang_acc, ang_acc_space.low, ang_acc_space.high)

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            previous_quat = Quaternion(self.quat[i])
            original_pos, original_quat = self.CLIENT.getBasePositionAndOrientation(self.DRONE_IDS[i])
            # _, orig_ang_v = self.CLIENT.getBaseVelocity(self.DRONE_IDS[i])
            # ang_v = np.array(orig_ang_v)
            # ang_v[0] = orig_ang_v[2]
            # ang_v[2] = orig_ang_v[0]

            quat = Quaternion(original_quat)
            
            current_ang_v = utils.intrinsic_euler_from_quats(previous_quat, quat, self.on_quaternion_error)*self.RAD2DEG/self.TIMESTEP
            ang_acc = (current_ang_v - self.obs["ang_v"][i])/self.TIMESTEP
            self._update_ang_acc(ang_acc)
            self._update_ang_v(current_ang_v)
            self.pos[i], self.quat[i] = np.array(original_pos), quat.q

    ################################################################################
    def _physics(self):
        """Base PyBullet physics implementation.
        """
        forces = (self.rpm**2)*self.KF
        torques = (self.rpm**2)*self.KM
        
        for i in range(self.NUM_DRONES):
            for m in range(4):
                motor_id = self.drone_to_link_name_to_index[i][f"motor{m}"]
                self.CLIENT.applyExternalForce(self.DRONE_IDS[i],
                                    motor_id,
                                    forceObj=[0, 0, forces[i,m]],
                                    posObj=[0,0,0],
                                    flags=p.LINK_FRAME,
                                    )
                self.CLIENT.applyExternalTorque(self.DRONE_IDS[i],
                                    motor_id,
                                    torqueObj=[0, 0, torques[i, m]],
                                    flags=p.LINK_FRAME,
                                    )

    ################################################################################

    def _drag(self,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        rpm = self.rpm[nth_drone, :]
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot, drag_factors*np.array(self.vel[nth_drone, :]))
        self.CLIENT.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            4,
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            )

    ################################################################################

    def _saveLastAction(self,
                        action
                        ):
        """Stores the most recent action into attribute `self.last_action`.
        """
        self.last_action = action
    
    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = self.CLIENT.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      )
            self.Y_AX[nth_drone] = self.CLIENT.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      )
            self.Z_AX[nth_drone] = self.CLIENT.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      )
    
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(self.URDF).getroot()
        properties = URDF_TREE.find("properties")
        M = sum(map(lambda node: float(node.get("value")),URDF_TREE.findall(".//mass")))
        L = float(properties.get('arm'))
        MAX_RPM = float(properties.get('max_rpm'))
        KF = float(properties.get('kf'))
        KM = np.array(list(map(float, properties.get('km').split(" "))))
        MAX_SPEED_KMH = float(properties.get('max_speed_kmh'))
        DRAG_COEFF_XY = float(properties.get('drag_coeff_xy'))
        DRAG_COEFF_Z = float(properties.get('drag_coeff_z'))
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        return M, L, MAX_RPM, KF, KM, MAX_SPEED_KMH, DRAG_COEFF
    
    ################################################################################
    
    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones((self.NUM_DRONES, 4)),
            high=np.ones((self.NUM_DRONES, 4)))
    
    ################################################################################

    def _init_obs_space(self):
        shape = (self.NUM_DRONES, 3)
        return spaces.Dict(ObsSpaces(
            ang_v = spaces.Box(
                low=np.tile(-300.0, shape),
                high=np.tile(300.0, shape),
            ),
            ang_acc = spaces.Box(
                low=np.tile(0.0, shape),
                high=np.tile(0.0, shape),
            )
        ))


    def _obs_space(self):
        shape = (self.NUM_DRONES, 3)
        return spaces.Dict(ObsSpaces(
            ang_v = spaces.Box(
                low=np.tile(-3000.0, shape),
                high=np.tile(3000.0, shape),
            ),
            ang_acc = spaces.Box(
                low=np.tile(-20000.0, shape),
                high=np.tile(20000.0, shape),
            )
        ))

    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.
        """
        return action*self.MAX_RPM

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
        obs, reward, done, info = super().step(action)
        self.setpoint = self._calculate_setpoint()
        full_obs = self.observation(obs)
        reward = self.reward(action/2.0 + 0.5, full_obs)
        return full_obs, reward, self.done(done, reward), info

    def reset(self):
        self.setpoint = self._calculate_setpoint()
        return self.observation(super().reset())

    def reward(self, action, obs) -> float:
        # ang_vel = self.obs["ang_v"]
        # setpoint = self.setpoint

        # dot = np.dot(ang_vel, setpoint)
        # ang_vel_size = np.linalg.norm(ang_vel)
        # setpoint_size = np.linalg.norm(setpoint)
        # length_sim = 100 / (100 + np.abs(ang_vel_size - setpoint_size))
        # if(ang_vel_size < 0.1 or setpoint_size < 0.1):
        #     cos_sim = 1
        # else:
        #     cos_sim = (dot/(np.linalg.norm(ang_vel) *
        #                np.linalg.norm(setpoint))+1.0)/2.0
        
        # return cos_sim*length_sim*(1.0/(1.0+np.sum(action)))
        normed_obs = (obs["state"]["ang_v"] - 3000.0)/6000.0
        normed_setpoint = (obs["setpoint"]["ang_v"] - 3000.0)/6000.0
        normed_closeness = tf.cast(1.0 - np.abs(normed_obs - normed_setpoint), tf.float32)
        # print(normed_closeness)
        setpoint_following = dfl.p_mean(dfl.smooth_constraint(normed_closeness, 0.7, 1.0), 0.0)
        # print(setpoint_following)
        # reward_dfl = dfl.Constraints(0.0, {
        #     "setpoint_following": setpoint_following,
        #     "action_smallness": dfl.p_mean(1.0-action, 0)**3.0
        # })
        # reward_val = dfl.dfl_scalar(reward_dfl)
        action_smallness = dfl.p_mean(1.0-action, 0)
        # print(action_smallness)
        # reward_val = dfl.implies(setpoint_following, action_smallness)
        # print(action, reward_val)
        return action_smallness*normed_closeness

    def done(self, done, reward):
        return done

    def observation(self, obs):
        return {"state": obs, "setpoint": self.setpoint}

    def _observationSpace(self):
        max_setpoint = np.ones(3)*1000
        obs_space = spaces.Dict({
            "state": self.env.observation_space,
            "setpoint": spaces.Dict({
                "ang_v": spaces.Box(low=-max_setpoint, high=max_setpoint),
                "ang_acc": spaces.Box(low=-max_setpoint, high=max_setpoint),
            })
        })
        return obs_space
    
        ################################################################################

    def _calculate_setpoint(self):
        """ calculates the desired goal at the current time
        """

        # ang_v = np.array([[signals.step((-100, 100))(self.simulation_time(), seed) for seed in [121,122,123]]])
        ang_v = np.array([0.0, 0, 0.0]) 
        return ({"ang_v": ang_v, "ang_acc": np.array([0.0, 0, 0.0])})


if __name__ == "__main__":
    from sd.envs.drone.assets import load_assets



    env = SetpointedAttitudeEnv(gui=True, test=True, urdf_path=load_assets.get_urdf_path("cf2x.urdf"))
    i = 0
    rw_sum = 0
    while(1):
        full_obs, reward, done, info = env.step(env.action_space.sample())
        rw_sum += reward
        if(i % 100 == 0):
            env.render()
        if(i % 500 == 0):
            print(rw_sum)
            rw_sum = 0
            env.reset()
        i+=1
