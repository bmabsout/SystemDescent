"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

def PDAttitudeControl(self,
    control_timestep,
    cur_rpy_rates,
    target_rpy_rates
    ):
    """DSL's CF2.x PID attitude control.

    Parameters
    ----------
    control_timestep : float
        The time step at which control is computed.
    cur_rpy_rates : ndarray
        (3,1)-shaped array of floats containing the current rotational rate
    target_rpy_rates : ndarray
        (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

    Returns
    -------
    ndarray
        (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

    """
    thrust = 20000.0
    rpy_rates_e = target_rpy_rates - cur_rpy_rates
    print(target_rpy_rates)
    print(cur_rpy_rates)
    #### PD target torques ####################################
    target_torques = - np.multiply(self.D_COEFF_TOR, rpy_rates_e)
    target_torques = np.clip(target_torques, -3200, 3200)
    pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
    print(pwm)
    pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
    action = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    return action

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Attitude control in pybullet')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=3,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 1.0
    H_STEP = .05
    R = .1*ARGS.num_drones
    INIT_XYZS = np.array([[R*np.cos((i/ARGS.num_drones)*2*np.pi+np.pi/2), R*np.sin((i/ARGS.num_drones)*2*np.pi+np.pi/2), H] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    env = CtrlAviary(drone_model=ARGS.drone,
                        num_drones=ARGS.num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=ARGS.physics,
                        neighbourhood_radius=10,
                        freq=ARGS.simulation_freq_hz,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        gui=ARGS.gui,
                        record=ARGS.record_video,
                        obstacles=ARGS.obstacles,
                        user_debug_gui=ARGS.user_debug_gui
                    )

    for drone_id, location in zip(env.DRONE_IDS, INIT_XYZS):
        cid = p.createConstraint(drone_id, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], location)
        # ball joint constrains the drone to a certain point in space
    

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    
    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0.0,0.0,0.0,0.00]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)] = PDAttitudeControl(ctrl[j],
                        cur_rpy_rates= obs[str(j)]["state"][13:16],
                        target_rpy_rates=np.array([100.0,0.0,0.0]),
                        control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                    )
                # action[str(j)], _, _ = ctrl[j].computeControlFromState(
                #     control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                #     state=obs[str(j)]["state"],
                #     target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                #     # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                #     target_rpy=INIT_RPYS[j, :]
                # )
                print(action)

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()


    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()
