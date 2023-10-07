import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from keras import layers
# import dill
from sd import utils

# @tf.function
def ball_differential_eq(states, actions):
    """
    ## Description

    The function describes the dynamic of the ball. It takes the state and the action as input and returns the next state.
    
    ## Coordinate and notation

        plate_rot_x, plate_rot_y          : the respective angle of the plane
        plate_vel_x, plate_vel_y          : the respective angular velocity of the plane
        ball_pos_x, ball_pos_y            : the position of the ball
        ball_vel_x, ball_vel_y            : the velocity of the ball
        m                                 : mass of the ball
        g                                 : gravity

    ## Math
        new_plate_rot_i =  plate_rot_i + plate_vel_i * dt 

        F * dt = m * dv
        mg * sin(plate_rot_i) * dt = m * dv_i

        new_v_i = g * sin(plate_rot_i) * dt + v_i
        new_s_i = new_v_i * dt + s_i               

    ## States Space

        The state is a tensor with shape (batch_size, 8). The latter dimension is the following
        right now the plate velocity is assumed to be constant

        | Index   | Observation        | Min | Max |
        |-------  |-------------       |-----|-----|
        | 0,1     | pl_rot_x,y         | -pi | pi  |
        | 2,3     | pl_vel_x,y         | -pi | pi  |
        | 4,5     | ba_pos_x,y         | -10 | 10  |
        | 6,7     | ba_vel_x,y         | -10 | 10  |

    ## Action Space

        The action is a tensor with shape (batch_size, 2). The latter dimension is the following

        | Index | Action      | Min | Max |
        |-------|------------ |-----|-----|
        | 0     | sp_rot_x    | -10 | 10  |
        | 1     | sp_rot_y    | -10 | 10  |

    ## Reuturn

        New states

    """

    g = tf.constant(10.0)
    m = tf.constant(1.0)
    dt = tf.constant(0.05)
    max_rot_x = tf.constant(np.pi)     ## max tilt of the plane
    max_rot_y = tf.constant(np.pi)     
    max_ball_pos_x = tf.constant(10.0)     ## the ball should be confined within the border
    max_ball_pos_y = tf.constant(10.0)
    pl_vel  = tf.constant(np.pi / 6)  

    pl_rot_x, pl_rot_y,  pl_vel_x, pl_vel_y, ba_pos_x, ba_pos_y, ba_v_x, ba_v_y = tf.split(states, num_or_size_splits=8, axis=1) 
    sp_x, sp_y = tf.split(actions, num_or_size_splits=2, axis=1)

    # clip_by_value actions
    sp_x = tf.clip_by_value(sp_x, -max_rot_x, max_rot_x)
    sp_y = tf.clip_by_value(sp_y, -max_rot_y, max_rot_y)

    # update plate angular velocity direction, vel is assumed to be constant
    # if the plate rot is within one update of the setpoint, set rot to setpoint, 
    # else, set the plate vel to the corresponding direction 
    dist = tf.abs(dt * pl_vel)
    pl_rot_x, pl_vel_x = tf.cond(dist >= tf.abs(sp_x - pl_rot_x), 
                                 lambda: (sp_x, 0.0), 
                                 lambda: (pl_rot_x, tf.sign(sp_x - pl_rot_x) * pl_vel))
    pl_rot_y, pl_vel_y = tf.cond(dist >= tf.abs(sp_y - pl_rot_y), 
                                 lambda: (sp_y, 0.0), 
                                 lambda: (pl_rot_y, tf.sign(sp_y - pl_rot_y) * pl_vel))

    # update plate rot -- step dt wrt angular velocity
    new_pl_rot_x = tf.clip_by_value(pl_rot_x + pl_vel_x * dt, -max_rot_x, max_rot_x)
    new_pl_rot_y = tf.clip_by_value(pl_rot_y + pl_vel_y * dt, -max_rot_y, max_rot_y)

    # update states
    new_v_x = g * tf.sin(new_pl_rot_x) * dt + ba_v_x
    new_v_y = g * tf.sin(new_pl_rot_y) * dt + ba_v_y
    new_pos_x = new_v_x * dt + ba_pos_x  
    new_pos_y = new_v_y * dt + ba_pos_y

    # if new pos is out of bound, set vel to 0
    new_v_x = tf.cond(tf.abs(new_pos_x) >= max_ball_pos_x, lambda: 0.0, lambda: new_v_x)
    new_v_y = tf.cond(tf.abs(new_pos_y) >= max_ball_pos_y, lambda: 0.0, lambda: new_v_y)

    # clip ball position
    new_pos_x = tf.clip_by_value(new_pos_x, -max_ball_pos_x, max_ball_pos_x)
    new_pos_y = tf.clip_by_value(new_pos_y, -max_ball_pos_y, max_ball_pos_y)

    new_state = tf.concat([new_pl_rot_x, new_pl_rot_y, pl_vel_x, pl_vel_y, new_pos_x, new_pos_y, new_v_x, new_v_y], axis=1)
    return new_state

def amazingball_diff_model():
    input_state = keras.Input(shape=(8,))
    input_action = keras.Input(shape=(2,))
    latent_input = keras.Input(shape=(0,))
    inputs = layers.Concatenate()([input_state, input_action, latent_input])
    outputs = layers.Lambda(lambda x: ball_differential_eq(x[:, 0:8], x[:, 8:10]))(inputs)
    model = keras.Model(inputs={"state": input_state, "action": input_action, "latent": latent_input}, outputs=outputs)
    model.summary()
    return model

if __name__ == "__main__":
    model = amazingball_diff_model()
    filepath = utils.random_subdir("models/AmazingBall-v0")
    utils.save_checkpoint(model=model, path=filepath, id=0, extra_objs={"ball_differential_eq": ball_differential_eq})
