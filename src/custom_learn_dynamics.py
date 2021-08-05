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

def generator_def(env: gym.Env, hidden_sizes:list[int]):
    obs_space = env.observation_space
    act_space = env.action_space
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError
    state_size = obs_space.shape[0]
    state_input = keras.Input(shape=(obs_space.shape[0],))
    action_input = keras.Input(shape=(act_space.shape[0],))
    latent_input = keras.Input(shape=(1,))
    
    dense = layers.Concatenate()([state_input, action_input, latent_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu", kernel_initializer='lecun_normal')(dense)
    low = np.array(obs_space.low)
    high = np.array(obs_space.high)
    outputs = layers.Dense(state_size, activation="sigmoid")(dense)*(high-low) + low
    model = keras.Model(inputs={"state": state_input, "action": action_input, "latent": latent_input}, outputs=outputs, name="system_identifier")
    model.summary()
    return model

@tf.function
def generator_2d_batch(generator, batch):
    batch_shape = tf.shape(batch["state"])[0:2]
    @tf.function
    def flatten_batch(b):
        return tf.reshape(b, shape=tf.concat([[-1],b.shape[2:]], axis=0))
    generated = generator(utils.map_dict(flatten_batch, batch))
    unflattened = tf.concat([batch_shape, tf.shape(generated)[1:]], axis=0)
    return tf.reshape(generated, shape=unflattened)


def gather_mini_batch(env: gym.Env, episode_size: int, policy=random_policy):
    def true_generator():
        obs = env.reset()
        done = False
        ep_len = 0
        while True:
            action = policy(obs, env.action_space)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            # env.render()
            # time.sleep(1e-3)
            yield {"state": prev_obs, "action": action, "next_state": obs}
            ep_len += 1
            if done or ep_len >= episode_size:
                obs = env.reset()
                prev_obs = obs
                ep_len = 0
    return true_generator

def discriminator_def(env: gym.Env, hidden_sizes:list[int], num_states: int):
    obs_space = env.observation_space
    act_space = env.action_space
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError
    state_size = obs_space.shape[0]

    states_input = keras.Input(shape=(num_states, obs_space.shape[0]))
    actions_input = keras.Input(shape=(num_states, act_space.shape[0]))
    next_states_input = keras.Input(shape=(num_states, obs_space.shape[0]))
    
    dense = layers.Concatenate()([states_input, actions_input, next_states_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu", kernel_initializer='lecun_normal')(dense)
    flattened = layers.Flatten()(dense)
    output = layers.Dense(1, activation="sigmoid")(flattened)
    model = keras.Model(inputs={"state": states_input, "action": actions_input, "next_state": next_states_input}, outputs=output, name="discriminator")
    model.summary()
    return model

@tf.function
def generate_fakes(generator, batch):
    latent_shape = batch["state"].shape[0:2] + [generator.input["latent"].shape[-1]]
    return ({
        **batch,
        "next_state": generator_2d_batch(generator, {
            **batch,
            "latent": tf.random.normal(shape=latent_shape)
        }),
    })

@tf.function
def dfls(generator, discriminator, real_batch):
    fake_batch = generate_fakes(generator, real_batch)
    discriminated = discriminator(fake_batch)
    generator_dfl = utils.DFL(0.0, {
        "fooled_amount": utils.p_mean(discriminated, 2.0)
    })
    discriminator_dfl = utils.DFL(0.0, {
        "real": utils.p_mean(discriminator(real_batch), 2.0),
        "fake": utils.p_mean(1 - discriminated, 2.0)
    })
    return generator_dfl, discriminator_dfl

def train_step(generator, discriminator, learning_rate):
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    @tf.function
    def gradient_step(batch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_dfl, discriminator_dfl = dfls(generator, discriminator, batch)
            generator_scalar = utils.dfl_scalar(generator_dfl)
            generator_loss = 1-generator_scalar
            discriminator_scalar = utils.dfl_scalar(discriminator_dfl)
            discriminator_loss = 1-discriminator_scalar
            discriminator_turn = generator_scalar > discriminator_scalar
            # scalar = discriminator_scalar if discriminator_turn else generator_scalar

        if discriminator_turn:
            grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
            optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))
        else:
            grads = gen_tape.gradient(generator_loss, generator.trainable_weights)
            optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        return (generator_scalar, generator_dfl), (discriminator_scalar, discriminator_dfl)

    def train_and_show(batch):
        generator_info, discriminator_info = gradient_step(batch)
        def show_info(scalar_dfl):
            scalar, dfl = scalar_dfl
            return f"Scalar: {scalar:.2e}|||{utils.format_dfl(dfl)}"
        return f"Generator: [{show_info(generator_info)}], Discriminator: [{show_info(discriminator_info)}]"
            

    return train_and_show

def system_identify(env_name: str, generator_hidden_sizes: list[int], discriminator_hidden_sizes: list[int], batch_size: int, num_states: int, num_batches: int, epochs: int, episode_size: int, num_validation_batches: int,  save_freq: int, learning_rate: float):
    env = gym.make(env_name)
    generator = generator_def(env, generator_hidden_sizes)
    discriminator = discriminator_def(env, discriminator_hidden_sizes, num_states)
    filepath = utils.random_subdir("models/" + env_name)
    dataset_spec = (
        { "state": tf.TensorSpec(shape=env.observation_space.shape,dtype=tf.float32)
        , "action": tf.TensorSpec(shape=env.action_space.shape, dtype=tf.float32)
        , "next_state": tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32)
        })

    dataset = tf.data.Dataset.from_generator(gather_mini_batch(env, episode_size), output_signature=dataset_spec) \
        .shuffle(episode_size*10).batch(num_states, drop_remainder=True).batch(batch_size, drop_remainder=True)
    
    validation_data = dataset.take(num_validation_batches).cache()
    data = dataset.take(num_batches).cache(filename='cache.tmp')
    utils.train_loop([data]*epochs, train_step(generator, discriminator, learning_rate), lambda epoch, dt: utils.save_checkpoint(filepath, generator, epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="gym environment", default="Pendulum-v0")
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_states', type=int, default=20)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--num_validation_batches', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--generator_hidden_sizes', nargs="+", type=int, default=[100,50, 4])
    parser.add_argument('--discriminator_hidden_sizes', nargs="+", type=int, default=[30,20, 4])
    args = parser.parse_args()
    system_identify(**vars(args))
