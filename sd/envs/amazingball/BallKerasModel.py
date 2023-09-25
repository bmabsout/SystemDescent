import tensorflow as tf
import numpy as np
import pybullet as p
import time
from tensorflow import keras
from keras import layers
# import dill
from sd import utils

def ball_differential_eq(states, actions):
    """
    ## Description

    The function describes the dynamic of the ball. It takes the state and the action as input and returns the next state.
    
    ## Coordinate and notation

        theta_x, theta_y: the respective angle of the plane
        s_x, s_y            : the position of the ball
        v_x, v_y            : the velocity of the ball
        m                   : mass of the ball
        g                   : gravity

    ## Math

        F * dt = m * dv
        mg * sin(theta_i) * dt = m * dv_i

        new_v_i = g * sin(theta_i) * dt + v_i
        new_s_i = v_i * dt + s_i              ## should use v_i or new_v_i 

    ## States Space

        The state is a tensor with shape (batch_size, 4). The latter dimension is the following

        | Index | Observation | Min | Max |
        |-------|-------------|-----|-----|
        | 0     | s_x         | -pi | pi  |
        | 1     | s_y         | -pi | pi  |
        | 2     | v_x         | -10 | 10  |
        | 3     | v_y         | -10 | 10  |

    ## Action Space

        The action is a tensor with shape (batch_size, 2). The latter dimension is the following

        | Index | Action     | Min | Max |
        |-------|------------|-----|-----|
        | 0     | theta_x    | -10 | 10  |
        | 1     | theta_y    | -10 | 10  |

    ## Reuturn

        New states

    """

    g = tf.constant(10.0)
    m = tf.constant(1.0)
    dt = tf.constant(0.05)
    max_thetax = tf.constant(np.pi)     ## max tilt of the plane
    max_thetay = tf.constant(np.pi)     
    max_ball_sx = tf.constant(10.0)     ## the ball should be confined within the border
    max_ball_sy = tf.constant(10.0)

    s_x, s_y, v_x, v_y = tf.split(states, num_or_size_splits=4, axis=1) 
    theta_x, theta_y = tf.split(actions, num_or_size_splits=2, axis=1)

    # clip actions
    theta_x = tf.clip_by_value(theta_x, -max_thetax, max_thetax)
    theta_y = tf.clip_by_value(theta_y, -max_thetay, max_thetay)

    # update states
    new_v_x = g * tf.sin(theta_x) * dt + v_x
    new_v_y = g * tf.sin(theta_y) * dt + v_y
    new_s_x = new_v_x * dt + s_x   # here to debate whether to use v_x or new_v_x, GPT call the curret semi-implicit Euler method
    new_s_y = new_v_y * dt + s_y

    # clip states
    new_s_x = tf.clip_by_value(new_s_x, -max_ball_sx, max_ball_sx)
    new_s_y = tf.clip_by_value(new_s_y, -max_ball_sy, max_ball_sy)

    new_state = tf.concat([new_s_x, new_s_y, new_v_x, new_v_y], axis=1)

    return new_state

ball_x = 0
plate_rx = 0

def get_sphere_position():
    # Replace with your logic
    global ball_x
    ball_x += 0.01
    return [ball_x, 0, 0.06]

def get_plate_rotation():
    global plate_rx
    plate_rx += 0.01
    return [plate_rx, 0, 0]  # Roll, Pitch, Yaw

if __name__=='__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath("sd/envs/amazingball/assets")
    plate = p.loadURDF("plate.urdf")

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.04)
    sphere = p.createMultiBody(baseVisualShapeIndex=visualShapeId)

    # Delta time (change this value based on your requirement)
    dt = 0.01

    try:
        while True:
            sphere_pos = get_sphere_position()
            plate_rot = get_plate_rotation()

            # Set sphere position
            p.resetBasePositionAndOrientation(sphere, sphere_pos, p.getQuaternionFromEuler([0, 0, 0]))

            # Set plate rotation
            p.resetBasePositionAndOrientation(plate, [0, 0, 0], p.getQuaternionFromEuler(plate_rot))

            time.sleep(dt)

    except KeyboardInterrupt:
        # Disconnect when script is interrupted
        p.disconnect()



        
