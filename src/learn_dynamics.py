import tensorflow as tf
# import neuroflight_trainer.gyms
import gym
import os.path as osp
import numpy as np
from gym.spaces import Box, Discrete
import argparse
import time
from tensorflow import keras
from tensorflow.keras import layers
import utils

def random_policy(obs, action_space):
    return action_space.sample()

def keras_model(env: gym.Env, hidden_sizes:list):
    obs_space = env.observation_space
    act_space = env.action_space
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError
    state_size = obs_space.shape[0]
    state_input = keras.Input(shape=(obs_space.shape[0],))
    action_input = keras.Input(shape=(act_space.shape[0],))
    
    dense = layers.Concatenate()([state_input, action_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu", kernel_initializer='lecun_normal')(dense)
    low = np.array(obs_space.low)
    high = np.array(obs_space.high)
    outputs = layers.Dense(state_size, activation="sigmoid")(dense)*(high-low) + low
    model = keras.Model(inputs=[state_input, action_input], outputs=outputs, name="system_identifier")
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



def system_identify(env_name: str, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int, learning_rate: float):
    env = gym.make(env_name)
    model = keras_model(env, hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate)
        , loss="mse")
    filepath = utils.random_subdir("models/" + env_name)
        # summary_writer = tf.summary.FileWriter(osp.join(filepath, "tensorboard"), graph=tf.get_default_graph())
    for batch_index in range(1, mini_batches+1):
        batch = gather_mini_batch(env, mini_batch_size, episode_size) #, policy=lambda o,a: np.array([0]))
        test_batch = gather_mini_batch(env, mini_batch_size//2, episode_size)
        model.fit(x=[batch["prev_states"], batch["actions"]], y=batch["states"], epochs=steps_per_batch, validation_data=([test_batch["prev_states"], test_batch["actions"]], test_batch["states"]), batch_size=2048)
        if batch_index % save_freq == 0:
            utils.save_checkpoint(filepath, model, batch_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="environment", default="Pendulum-v0")
    parser.add_argument('--mini_batches', type=int, default=400)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--mini_batch_size', type=int, default=100000)
    parser.add_argument('--steps_per_batch', type=int, default=100)
    parser.add_argument('--hidden_sizes', nargs="+", type=int, default=[256,256])
    args = parser.parse_args()
    system_identify(**vars(args))
