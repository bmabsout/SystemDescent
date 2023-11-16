import argparse
import datetime
from pathlib import Path
from sd.envs.amazingball.BallKerasModel import amazingball_diff_model
from sd.envs.amazingball.ModeledAmazingBall import ModeledAmazingBall
from sd.lyapunov import V_def, actor_def
from sd import utils
from typing import Tuple
from sd.envs.amazingball.abs_utils import flatten_system_state, flatten_setpoint, get_ballstate, get_setpoint
from tensorflow import keras
import numpy as np
import tensorflow as tf
import gymnasium as gym

def generate_dataset(env: gym.Env):
    def gen_sample():
        """Generates a sample from the environment but assumes that the environment is abs"""
        while True:
            _, _ = env.reset()
            yield {"system_state": env.env.state, "setpoint":{"ball_pos": [0.0, 0.0], "ball_vel": [0.0, 0.0]}} 
    return gen_sample

@tf.function
def step_through(states, setpoints, dynamic, actor, nsteps):
    """     step through [states] for [nsteps] times with fixed [setpoints] 
            using the model provided
            
            return the [final_states] and accumulated intermediate states [acc_states] 
    """
    states = flatten_system_state(states)
    setpoints = flatten_setpoint(setpoints)

    acc_states = tf.TensorArray(tf.float32, size=nsteps+1)
    acc_actions = tf.TensorArray(tf.float32, size=nsteps)
    current_states = states
    acc_states = acc_states.write(0, current_states)
    for i in range(nsteps):
        latent_shape = tuple(current_states.shape[0:1]) + tuple(dynamic.input["latent"].shape[1:])
        action = actor({"state":current_states, "setpoint":setpoints})

        current_states = dynamic(
            { 
                "plate_rot": current_states[:,0:2], 
                "plate_vel": current_states[:,2:4], 
                "ball_pos": current_states[:,4:6], 
                "ball_vel": current_states[:,6:8],
                "action": action,
                "latent": tf.random.normal(latent_shape)
            }, training=True
        ) 

        current_states = flatten_system_state(current_states)
        acc_states = acc_states.write(i+1, current_states)
        acc_actions = acc_actions.write(i, action)

    tf_conformed_states = tf.transpose(acc_states.stack(), [1,0,2])  ## put batch dim in front shape=(batch, nsteps, state_dim)
    tf_conformed_actions = tf.transpose(acc_actions.stack(), [1,0,2])  ## put batch dim in front shape=(batch, nsteps, action_dim)
    return current_states, tf_conformed_states, tf_conformed_actions


def save_abs_actor(model):
    path = Path(utils.latest_model()).parent / 'actor.keras'
    model.save(path)


def train():
    batch_size = 64

    env = ModeledAmazingBall(render_mode="human")   
    action_shape        = env.action_space.shape                               # (2,)   
    system_state_shape  = (8,) 
    setpoint_shape      = ball_state_shape          = (4,)                            # ball pos and vel
    actor_input_shape   = (system_state_shape[0] + setpoint_shape[0],)    # (12,)

    dynamics_model = amazingball_diff_model()
    # dynamics_model = utils.load_checkpoint(utils.latest_model()) 
    actor = actor_def(system_state_shape, action_shape, input_setpoint_shape = setpoint_shape)
    # lyapunov_model = V_def(set)   
    # lyapunov_input_shape = () 

    dataset_spec = { 
                    "system_state": 
                                {
                                    "plate_rot": tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                    "plate_vel": tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                    "ball_pos": tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                    "ball_vel": tf.TensorSpec(shape=(2,), dtype=tf.float32)
                                },
                    "setpoint": 
                                {
                                    "ball_pos": tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                    "ball_vel": tf.TensorSpec(shape=(2,), dtype=tf.float32)
                                }
                    }   
    dataset = tf.data.Dataset.from_generator(generate_dataset(env), output_signature=dataset_spec)  
    batched_dataset = dataset.batch(batch_size).take(500).cache().repeat(10)

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    train_summary_write, _ = start_logging()
    for i,batch in enumerate(batched_dataset):
        with tf.GradientTape() as tape:
            # rd is a tensorflow random int
            cur_min = 5.0 + (i * 45) / 5000  
            rd = tf.random.uniform(shape=(), minval=int(cur_min), maxval=100, dtype=tf.int32)
            fin_states, acc_states, acc_actions = step_through(batch['system_state'], batch['setpoint'], dynamics_model, actor, rd)
            flat_ballstate = get_ballstate(fin_states, format="flat")
            # flat_ballstate = get_ballstate(acc_states, format="flat")
            flat_setpoint = get_setpoint(batch['setpoint'], format='flat')
            # distances = tf.norm(flat_ballstate - flat_setpoint, ord='euclidean', axis=1)
            distances = tf.norm(flat_ballstate[:,:2] - flat_setpoint[:,:2], ord='euclidean', axis=1)
            action_sizes =  tf.sqrt(tf.reduce_mean(acc_actions**2.0))

            loss = tf.reduce_mean(distances) + 2*action_sizes

        grads = tape.gradient(loss, actor.trainable_variables)
        optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    save_abs_actor(actor)


def start_logging():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer
	
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-init-ballstate', type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0], help='initial ball state')
    args = parser.parse_args()

    if args.train:
        train()
        exit(0)
    
    actor = keras.models.load_model( "models/AmazingBall-v0/730715/checkpoints/checkpoint0/actor.keras")
    setpoints = np.array([[0.0, 0.0, 0.0, 0.0]])

    env = ModeledAmazingBall(render_mode="human")
    obs, info = env.reset()
    while(1):
        # flat_system_states = flatten_system_state(env.state)   # shape = (8,)
        flat_system_states = np.expand_dims(obs, axis=0)  # shape = (1,8)
        action = actor({"state":flat_system_states, "setpoint":setpoints})
        spx, spy = action[0,0], action[0,1]
        obs, reward, done, truncated, info = env.step(np.array([spx, spy]))

    










