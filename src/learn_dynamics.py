import tensorflow as tf
# import neuroflight_trainer.gyms
import gym
import os.path as osp
from pathlib import Path
import numpy as np
from gym.spaces import Box, Discrete
import argparse
import uuid
#import KerasPendulum
import time
from tensorflow import keras
from tensorflow.keras import layers
import utils

def random_policy(obs, action_space):
    return action_space.sample()

def keras_model(env: gym.Env, hidden_sizes:list):
    if(not (isinstance(env.action_space, gym.spaces.Box) and isinstance(env.action_space, gym.spaces.Box))):
        raise NotImplementedError
    state_size = env.observation_space.shape[0]
    state_input = keras.Input(shape=(env.observation_space.shape[0],))
    action_input = keras.Input(shape=(env.action_space.shape[0],))
    
    dense = layers.Concatenate()([state_input, action_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu", kernel_initializer='lecun_normal')(dense)
    outputs = layers.Dense(state_size)(dense)
    model = keras.Model(inputs=[state_input, action_input], outputs=outputs, name="system_indentifier")
    model.summary()
    return model


def gather_mini_batch(env: gym.Env, mini_batch_size: int, episode_size: int, policy=random_policy):
    obs = env.reset()
    done = False
    ep_len = 0
    total_steps_in_batch = 0
    prev_states, actions, states = [], [], []
    for i in range(mini_batch_size):
        action = policy(obs, env.action_space)
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        # env.render()
        # time.sleep(1e-3)
        prev_states.append(prev_obs)
        actions.append(action)
        states.append(obs)
        ep_len += 1
        total_steps_in_batch += 1
        if done or ep_len >= episode_size:
            obs = env.reset()
            prev_obs = obs
            ep_len = 0


    shuffled_indices = np.arange(len(prev_states))
    np.random.shuffle(shuffled_indices)
    return {
        "prev_states": np.array(prev_states)[shuffled_indices],
        "actions": np.array(actions)[shuffled_indices],
        "states": np.array(states)[shuffled_indices]
    }


def system_identify(env: gym.Env, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int, learning_rate: float):
    model = keras_model(env, hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate)
        , loss="mse")
    uniq_id = uuid.uuid1().__str__()[:6]
    filepath = "saved/" + uniq_id
        # summary_writer = tf.summary.FileWriter(osp.join(filepath, "tensorboard"), graph=tf.get_default_graph())
    for batch_index in range(1, mini_batches+1):
        batch = gather_mini_batch(env, mini_batch_size, episode_size) #, policy=lambda o,a: np.array([0]))
        test_batch = gather_mini_batch(env, mini_batch_size//2, episode_size)
        model.fit(x=[batch["prev_states"], batch["actions"]], y=batch["states"], epochs=steps_per_batch, validation_data=([test_batch["prev_states"], test_batch["actions"]], test_batch["states"]), batch_size=2048)
        if batch_index % save_freq == 0:
            checkpoint_name = "checkpoint{}".format(batch_index)
            checkpoint_path = Path(filepath, "checkpoints", checkpoint_name)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            print("saving: ", str(checkpoint_path))
            model.save(str(checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="environment", default="Pendulum-v0")
    parser.add_argument('--mini_batches', type=int, default=400)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--mini_batch_size', type=int, default=100000)
    parser.add_argument('--steps_per_batch', type=int, default=100)
    parser.add_argument('--hidden_sizes', nargs="+", type=int, default=[256,256])
    args = parser.parse_args()
    args.env = gym.make(args.env)
    system_identify(**vars(args))
    # gather_mini_batch(args.env, args.mini_batch_size)
