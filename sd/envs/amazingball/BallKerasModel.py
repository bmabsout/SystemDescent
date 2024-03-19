import tensorflow as tf
import numpy as np
import keras
from sd import utils
from sd.envs.amazingball.constant import constants

@keras.saving.register_keras_serializable(package="sd.envs.amazingball", name="ball_differential_eq")
# @tf.function
def ball_differential_eq(constants, input):
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
    states = input['state']
    actions = input['action']
    g = constants['g']
    m = constants['m']
    dt = constants['dt']
    max_rot_x = constants['max_rot_x']
    max_rot_y = constants['max_rot_y']
    max_ball_pos_x = constants['max_ball_pos_x']
    max_ball_pos_y = constants['max_ball_pos_y']
    pl_vel  = constants['pl_vel']
    max_ball_vel  = constants['max_ball_vel']
    collision_damping = constants['collision_damping']
    pl_rot_x, pl_rot_y,  pl_vel_x, pl_vel_y, ba_pos_x, ba_pos_y, ba_v_x, ba_v_y = tf.split(states, num_or_size_splits=8, axis=1) 

    sp_x, sp_y = tf.split(actions, num_or_size_splits=2, axis=1)

    # clip_by_value actions
    sp_x = tf.clip_by_value(sp_x, -max_rot_x, max_rot_x)
    sp_y = tf.clip_by_value(sp_y, -max_rot_y, max_rot_y)

    # update plate angular velocity direction, vel is assumed to be constant
    # if the plate rot is within one update of the setpoint, set rot to setpoint, 
    # else, set the plate vel to the corresponding direction 
    dist = tf.abs(dt * pl_vel)

    # For pl_rot_x and pl_vel_x
    condition_x = dist >= tf.abs(sp_x - pl_rot_x)
    pl_rot_x = tf.where(condition_x, sp_x, pl_rot_x)
    pl_vel_x = tf.where(condition_x, pl_vel_x * 0.0, tf.sign(sp_x - pl_rot_x) * pl_vel)

    condition_y = dist >= tf.abs(sp_y - pl_rot_y)
    pl_rot_y = tf.where(condition_y, sp_y, pl_rot_y)
    pl_vel_y = tf.where(condition_y, pl_vel_y * 0.0, tf.sign(sp_y - pl_rot_y) * pl_vel)

    # update plate rot -- step dt wrt angular velocity
    new_pl_rot_x = tf.clip_by_value(pl_rot_x + pl_vel_x * dt, -max_rot_x, max_rot_x)
    new_pl_rot_y = tf.clip_by_value(pl_rot_y + pl_vel_y * dt, -max_rot_y, max_rot_y)

    # update states
    new_v_x = tf.clip_by_value(g * tf.sin(new_pl_rot_x) * dt + ba_v_x, -max_ball_vel, max_ball_vel)
    new_v_y = tf.clip_by_value(g * tf.sin(new_pl_rot_y) * dt + ba_v_y, -max_ball_vel, max_ball_vel)

    new_pos_x = new_v_x * dt + ba_pos_x  
    new_pos_y = new_v_y * dt + ba_pos_y

    # handle collision
    condition_x = tf.abs(new_pos_x) >= max_ball_pos_x
    condition_y = tf.abs(new_pos_y) >= max_ball_pos_y

    # collision handle 1: if new pos is out of bound, set vel to 0
    # new_v_x = tf.where(condition_x, 0.0, new_v_x)
    # new_v_y = tf.where(condition_y, 0.0, new_v_y)
    # collision handle 1: END

    # collision handle 2: if new pos is out of bound, set vel to -vel
    new_v_x = tf.where(condition_x, -new_v_x * collision_damping, new_v_x)
    new_v_y = tf.where(condition_y, -new_v_y * collision_damping, new_v_y)
    # collision handle 2: END

    # clip ball position
    new_pos_x = tf.clip_by_value(new_pos_x, -max_ball_pos_x, max_ball_pos_x)
    new_pos_y = tf.clip_by_value(new_pos_y, -max_ball_pos_y, max_ball_pos_y)
    # tf.print("new_state:",[new_pl_rot_x, new_pl_rot_y, pl_vel_x, pl_vel_y, new_pos_x, new_pos_y, new_v_x, new_v_y])
    new_state = tf.concat([new_pl_rot_x, new_pl_rot_y, pl_vel_x, pl_vel_y, new_pos_x, new_pos_y, new_v_x, new_v_y], axis=1)
    return new_state

@keras.saving.register_keras_serializable(package="CustomLayers", name="DiffEqLayer")
class DiffEqLayer(keras.layers.Layer):
    def __init__(self, constants):
        super().__init__()
        self.tf_constants = utils.map_dict_elems(lambda x: np.array(x, dtype=np.float32), constants)

    def call(self, inputs):
        return ball_differential_eq(self.tf_constants, inputs)

    def get_config(self):
        return {"constants": self.tf_constants}

def amazingball_diff_model():
    inputs = {
        "state": keras.Input(shape=(8,), name="state"),
        "action": keras.Input(shape=(2,), name="action"),
        "latent": keras.Input(shape=(0,), name="latent")
    }
    
    model = keras.Model(inputs=inputs, outputs=DiffEqLayer(constants)(inputs))
    return model

if __name__ == "__main__":
    model = amazingball_diff_model()
    filepath = utils.random_subdir("models/AmazingBall-v0")
    # pickle.dump(model, open("chompe.pd", "wb"))
    # m = pickle.load(open("chompe.pd", "rb"))
    # m.
    utils.save_checkpoint(model=model, path=filepath, id=0, extra_objs={"ball_differential_eq": ball_differential_eq})
