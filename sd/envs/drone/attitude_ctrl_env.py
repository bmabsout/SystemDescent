"""
Adapted from 
"""


import os
from sys import platform
import time
import collections
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as etxml
import numpy as np
from pybullet_utils import bullet_client as bc
import pybullet as p
import pybullet_data
import gym
from gym import core, spaces
import sd.envs.drone.signals as signals
from sd.envs.drone.assets import load_assets
from scipy.spatial.transform import Rotation
from gym.envs.registration import register
from sd.rl.utils import SqueezeWrapper



def SqueezedAttitudeControlEnv(**kwargs):
    return SqueezeWrapper(AttitudeControlEnv(**kwargs))

class AttitudeControlEnv(gym.Env):

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

        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self.first_render_call = True
        self._housekeeping()
        self._initialize_pybullet()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
    
    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        # self.CLIENT.resetSimulation()
        #### Housekeeping ##########################################
        self._housekeeping()
        for droneId, i in zip(self.DRONE_IDS, range(len(self.DRONE_IDS))):
            self.CLIENT.resetBaseVelocity(droneId, self.vel[i], self.ang_v[i])
            self.CLIENT.resetBasePositionAndOrientation(droneId, self.INIT_XYZS[i], self.quat[i])
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Return the initial observation ########################
        return self._computeObs()
    
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
        if self.GUI and self.USER_DEBUG:
            current_input_switch = self.CLIENT.readUserDebugParameter(self.INPUT_SWITCH)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = not self.USE_GUI_RPM
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = self.CLIENT.readUserDebugParameter(int(self.SLIDERS[i]))
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
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
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            self._saveLastAction(action)
            clipped_action = self._preprocessAction(action)
        self.rpm = clipped_action
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                self._physics(i)
                # _drag(i)
            self.CLIENT.stepSimulation()
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        self.setpoint = self._calculate_setpoint()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward(self.ang_v, self.setpoint, action)
        done = self._computeDone()
        done = self._simulation_time() > reward*5
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info
    
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
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self._simulation_time(), self.SIM_FREQ, (self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print(f"[INFO] BaseAviary.render() ——— drone {i}",
                  f"——— xyz {np.array2string(self.pos[i], precision=2)}",
                  f"——— velocity {np.array2string(self.vel[i], precision=2)}",
                  f"——— rpy {np.array2string(self.rpy[i], precision=2)}",
                  f"——— ang_v {np.array2string(self.ang_v[i], precision=2)}",
                  f" ———  rpm {np.array2string(self.rpm[i], precision=2)}")
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        self.CLIENT.disconnect()
    
    ################################################################################

    def getPyBulletClient(self):
        """
        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """
        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
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
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE
                                              ) for i in range(self.NUM_DRONES)])
        
        self.drone_to_link_name_to_index = []

        for drone_id, location in zip(self.DRONE_IDS, self.INIT_XYZS):
            cid = self.CLIENT.createConstraint(drone_id, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], location)

            self.drone_to_link_name_to_index.append({self.CLIENT.getBodyInfo(drone_id):-1,})
        
            for _id in range(p.getNumJoints(drone_id)):
                _name = p.getJointInfo(drone_id, _id)[12].decode('UTF-8')
                self.drone_to_link_name_to_index[-1][_name] = _id

        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        self.debug_force_line = -np.ones(4)

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.setpoint = self._calculate_setpoint()
        
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES, 4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (1, self.NUM_DRONES))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))

    
    ################################################################################


    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            pos, quat = self.CLIENT.getBasePositionAndOrientation(self.DRONE_IDS[i])
            change = Rotation.from_quat(self.quat[i]).inv() * Rotation.from_quat(quat)
            intrinsic_euler = change.as_euler('XYZ', degrees=True)/self.TIMESTEP
            self.ang_v[i] = intrinsic_euler
            self.pos[i], self.quat[i] = pos, quat

            # self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            # print(diff.as_euler('XYZ'))
            # orn = p.getQuaternionFromEuler([roll, pitch, yaw])
            # p.resetBasePositionAndOrientation(self.DRONE_IDS[i], self.pos[i], orn)
            
        # self.rpy *= self.RAD2DEG
    ################################################################################
    def _physics(self,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        rpm = self.rpm[nth_drone, :]
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        
        for i in range(4):
            motor_id = self.drone_to_link_name_to_index[nth_drone][f"motor{i}"]
            # motor_id = i+1
            # self.debug_force_line[i] = p.addUserDebugLine(
            #     lineFromXYZ=[0, 0, 0],
            #     lineToXYZ=[0, 0, forces[i]],
            #     lineColorRGB=[1, 1, 0],
            #     parentObjectUniqueId=self.DRONE_IDS[nth_drone],
            #     parentLinkIndex=motor_id,
            #     replaceItemUniqueId=int(self.debug_force_line[i]),
            #     physicsClientId=self.CLIENT
            # )
            self.CLIENT.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 motor_id,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0,0,0],
                                 flags=p.LINK_FRAME,
                                 )
            self.CLIENT.applyExternalTorque(self.DRONE_IDS[nth_drone],
                                motor_id,
                                torqueObj=[0, 0, torques[i]],
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
        self.CLIENT.applyExternalForce(self.DRONE_IDS[nth_drone],
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
        box = spaces.Box(low=np.zeros((self.NUM_DRONES,4)), high=np.ones((self.NUM_DRONES, 4)))
        print("action space:", box)
        return box
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        max_angular_rate = np.ones((self.NUM_DRONES, 3))*10000
        max_setpoint = np.ones((self.NUM_DRONES, 3))*1000
        min_action = np.zeros((self.NUM_DRONES, 4))
        max_action = np.ones((self.NUM_DRONES, 4))
        low = np.concatenate([-max_angular_rate, min_action, -max_setpoint], axis=1)
        high = np.concatenate([max_angular_rate, max_action, max_setpoint], axis=1)
        box = spaces.Box(low, high)
        # box = spaces.Dict({i: spaces.Box(low=low[i],high=high[i]) for i in range(self.NUM_DRONES)})
        print("obs space:", box)
        return box
    
    ################################################################################
    
    def _computeObs(self):
        return np.concatenate([self.ang_v, self.last_action, self.setpoint], axis=1)
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        return action*self.MAX_RPM

    def _simulation_time(self):
        return self.step_counter*self.TIMESTEP

    ################################################################################

    def _calculate_setpoint(self):
        """ calculates the desired goal at the current time
        """

        # return np.array([[signals.step((-100, 100))(self._simulation_time(), seed) for seed in [121,122,123]]])
        return np.array([[0, 100, 0]])

    ################################################################################
    @staticmethod
    def _computeReward(ang_vel, setpoint, action):
        # print(ang_vel, setpoint)
        components = (100/(100+np.abs(ang_vel - setpoint)))
        p = 0.1
        
        reward = np.mean(components**p)**(1/p)
        dot = np.dot(ang_vel.squeeze(), setpoint.squeeze())
        ang_vel_size = np.linalg.norm(ang_vel)
        setpoint_size = np.linalg.norm(setpoint)
        length_sim = 100/ (100 + np.abs(ang_vel_size - setpoint_size))
        if(ang_vel_size < 0.1 or setpoint_size < 0.1):
            cos_sim = 1
        else:
            cos_sim = (dot/(np.linalg.norm(ang_vel)*np.linalg.norm(setpoint))+1.0)/2.0
        # print(components, reward, -np.mean(np.abs(ang_vel - setpoint)), dot, cos_sim, length_sim)
        # print(reward)
        # return -np.mean(np.abs(ang_vel - setpoint))

        return cos_sim*length_sim
        # return reward

    ################################################################################

    def _computeDone(self):
        return False

    ################################################################################

    def _computeInfo(self):
        return {}
        

if __name__ == "__main__":
    import importlib.resources as pkg_resources
    from sd.envs.drone.assets import load_assets

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


    env = AttitudeControlEnv(gui=True, urdf_path=load_assets.get_urdf_path("cf2x.urdf"))
    i = 0
    START = time.time()
    while(1):
        env.step(np.array([[0.0, 0.0, 0.0, 0.0]]))
        sync(i, START, env.TIMESTEP)
        if(i % 100 == 0):
            env.render()
        i+=1
