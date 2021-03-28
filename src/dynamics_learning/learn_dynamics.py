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
        dense = layers.Dense(hidden_size, activation="relu")(dense)
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


def system_identify(env: gym.Env, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int):
    model = keras_model(env, hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001)
        , loss="mse")
    uniq_id = uuid.uuid1().__str__()[:6]
    filepath = "saved/" + uniq_id
        # summary_writer = tf.summary.FileWriter(osp.join(filepath, "tensorboard"), graph=tf.get_default_graph())
    for batch_index in range(0, mini_batches):
        batch = gather_mini_batch(env, mini_batch_size, episode_size) #, policy=lambda o,a: np.array([0]))
        model.fit(x=[batch["prev_states"], batch["actions"]], y=batch["states"], steps_per_epoch=steps_per_batch, batch_size=mini_batch_size)
        path = osp.join(filepath, "checkpoints")
        if batch_index % save_freq == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
            filename = path+"/checkpoint{}.tf".format(batch_index)
            print("saving: ", filename)
            model.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="environment", default="Pendulum-v0")
    parser.add_argument('--mini_batches', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--episode_size', type=int, default=400)
    parser.add_argument('--mini_batch_size', type=int, default=10000)
    parser.add_argument('--steps_per_batch', type=int, default=100)
    parser.add_argument('--hidden_sizes', nargs="+", type=int, default=[128,256,128])
    args = parser.parse_args()
    args.env = gym.make(args.env)
    system_identify(**vars(args))
    # gather_mini_batch(args.env, args.mini_batch_size)
