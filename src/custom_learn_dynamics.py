import tensorflow as tf
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

def model_def(env: gym.Env, hidden_sizes:list):
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
    model = keras.Model(inputs={"state": state_input, "action": action_input}, outputs=outputs, name="system_identifier")
    model.summary()
    return model


def gather_mini_batch(env: gym.Env, episode_size: int, policy=random_policy):
    def generator():
        obs = env.reset()
        done = False
        ep_len = 0
        while True:
            action = policy(obs, env.action_space)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            # env.render()
            # time.sleep(1e-3)
            yield {"state": prev_obs, "action": action}, obs
            ep_len += 1
            if done or ep_len >= episode_size:
                obs = env.reset()
                prev_obs = obs
                ep_len = 0
    return generator

def system_identify(env_name: str, hidden_sizes: list, batch_size: int, num_batches: int, epochs: int, episode_size: int, num_validation_batches: int,  save_freq: int, learning_rate: float):
    env = gym.make(env_name)
    model = model_def(env, hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate)
        , loss="mse")
    filepath = utils.random_subdir("models/" + env_name)
    dataset_spec = (
        { "state": tf.TensorSpec(shape=env.observation_space.shape,dtype=tf.float32)
        , "action": tf.TensorSpec(shape=env.action_space.shape, dtype=tf.float32)
        }, tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(gather_mini_batch(env, episode_size), output_signature=dataset_spec) \
        .shuffle(episode_size*10).batch(batch_size)
    
    validation_data = dataset.take(num_validation_batches).cache()
    data = dataset.take(num_batches).cache()
    for epoch in range(1, epochs+1):
        model.fit(x=data, validation_data=validation_data, epochs=1)
        if epoch % save_freq == 0:
            utils.save_checkpoint(filepath, model, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="gym environment", default="Pendulum-v0")
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--num_validation_batches', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_sizes', nargs="+", type=int, default=[256,256])
    args = parser.parse_args()
    system_identify(**vars(args))
