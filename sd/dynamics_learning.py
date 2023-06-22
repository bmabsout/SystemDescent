from sd.envs import modelable_env
import tensorflow as tf
import gymnasium as gym
import os.path as osp
import numpy as np
from gymnasium.spaces import Box, Discrete
import argparse
import time
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
from . import utils
from sd import dfl
from typing import Union
import sd.envs # brings envs in scope
from sd.envs.modelable_env import ModelableEnv, ModelableWrapper

"""
To predict the dynamics. 
The decretized version of differential equation. 

Modeling the transition function (.update .step)
"""

def random_policy(obs, action_space):
    return action_space.sample()

def generator_def(env: ModelableEnv, hidden_sizes: list[int], latent_size:int):
    """
    The function return an untrained DNN representing the environment dynamics. 
    Lyapunov function is not included in the model, and it should not.

    Args:
        env (ModelableEnv): The environment to be modeled. This includes the observation space and action space.
        hidden_sizes (list[int]): The number of hidden layers and the number of neurons in each layer.
        latent_size (int): The dimension of the latent space. # Q: What is the latent space?

    Returns:
        keras.Model: The untrained DNN representing the environment dynamics.
    """
    obs_space = env.observation_space
    act_space = env.action_space
    print(act_space, obs_space)
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError

    state_size = obs_space.shape[0]
    state_input = keras.Input(shape=(obs_space.shape[0],))
    normalized_state = (state_input - obs_space.low)/(obs_space.high - obs_space.low)
    action_input = keras.Input(shape=(act_space.shape[0],))
    normalized_action = (action_input - act_space.low)/(act_space.high - act_space.low)
    latent_input = keras.Input(shape=(latent_size,))

    dense = layers.Concatenate()([normalized_state, normalized_action, latent_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu",
                             kernel_initializer='lecun_normal',
                             kernel_regularizer=utils.PMean())(dense)
    

    dense = layers.Dense(state_size)(dense)
    outputs = layers.Activation("sigmoid")(dense)*(obs_space.high-obs_space.low)+obs_space.low
    model = keras.Model(
        inputs={
            "state": state_input,
            "action": action_input,
            "latent": latent_input
        },
        outputs=outputs,
        name="system_identifier"
    )
    model.summary()
    return model

@tf.function
def generator_2d_batch(generator, batch):
    batch_shape = tf.shape(batch["state"])[0:2]

    @tf.function
    def flatten_batch(b):
        return tf.reshape(b, shape=tf.concat([[-1], b.shape[2:]], 0))

    generated = generator(utils.map_dict(flatten_batch, batch))
    unflattened = tf.concat([batch_shape, tf.shape(generated)[1:]], 0)
    return tf.reshape(generated, shape=unflattened)


def gather_mini_batch(env: ModelableEnv, episode_size: int, policy=random_policy):
    """
    Return a python generator that generates a series of state-action-next_state tuples.
    The trajectory is of size episode_size. The true_generator generates a 1_D array. 
    """
    def true_generator():
        obs, _ = env.reset()
        done = False
        ep_len = 0
        while True:
            action = policy(obs, env.action_space)
            prev_obs = obs
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            # time.sleep(1e-3)
            yield {"state": prev_obs, "action": action, "next_state": obs}
            ep_len += 1
            if ep_len >= episode_size:
                obs = env.reset()
                prev_obs = obs
                ep_len = 0
    return true_generator

def discriminator_def(env: ModelableEnv, hidden_sizes: list[int], num_transitions: int):
    """ Return model. When the model performs well. 
        It discriminates whether the transition originated from the environment (output=1) or the generator (output=0)."""
    obs_space = env.observation_space
    act_space = env.action_space
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError
    state_size = obs_space.shape[0]

    states_input = keras.Input(shape=(num_transitions, obs_space.shape[0]))
    actions_input = keras.Input(shape=(num_transitions, act_space.shape[0]))
    next_states_input = keras.Input(shape=(num_transitions, obs_space.shape[0]))

    dense = layers.Concatenate()([states_input, actions_input, next_states_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size,
                             activation="selu",
                             kernel_initializer='lecun_normal',
                             kernel_regularizer=utils.PMean())(dense)
    flattened = layers.Flatten()(dense)
    output = layers.Dense(1, activation="sigmoid")(flattened)
    model = keras.Model(
        inputs={
            "state": states_input,
            "action": actions_input,
            "next_state": next_states_input
        },
        outputs=output,
        name="discriminator"
    )
    model.summary()
    return model

@tf.function
def generate_fakes(generator, batch):
    latent_shape = tuple(batch["state"].shape[0:2]) + tuple(generator.input["latent"].shape[1:])
    return ({
        "state": batch["state"],
        "action": batch["action"],
        "next_state": generator_2d_batch(generator, {
            **batch,
            "latent": tf.random.normal(shape=latent_shape)
        }),
    })


@tf.function
def direct_dfls(generator, batch, closeness):
    return dfl.Constraints(1.0, {
        "closeness": closeness(generate_fakes(generator, batch)[
            "next_state"], batch["next_state"])
        # "reg": 1 - tf.tanh(tf.reduce_mean(generator.losses)*10.0)
    })


@tf.function
def gan_dfls(generator, discriminator, real_batch, closeness_dfl):
    fakes = generate_fakes(generator, real_batch)
    fakes2 = generate_fakes(generator, real_batch)
    fooled = dfl.p_mean(discriminator(fakes), 2.0)
    fooled2 = dfl.p_mean(discriminator(fakes2), 2.0)
    discriminator_regularizer = 1 - tf.tanh(tf.reduce_mean(discriminator.losses)*10.0)
    generator_regularizer = 1 - tf.tanh(tf.reduce_mean(generator.losses)*10.0)
    real_fake_dfl = dfl.Constraints(0.0, {
        "real": dfl.p_mean(discriminator(real_batch), 0.0),
        "fake": dfl.p_mean(1 - discriminator(fakes), 0.0)
    })
    fooled2_normalized = dfl.smooth_constraint(fooled2, 0.0, 0.5, 0.0, 0.97, starts_linear=True)
    generator_dfl = dfl.Constraints(2.0, {
#         "fooled": dfl.p_mean(discriminator(fakes2),2.0)
        "fooled": fooled2_normalized,
        # "direct": direct_dfls(generator, real_batch, closeness_dfl)
        # "reg": tf.where(tf.stop_gradient(dfl.dfl_scalar(real_fake_dfl)) < 0.1, generator_regularizer, 1.0)
    })
    generator_badness = 1.0 - tf.minimum(tf.stop_gradient(fooled2)*2, 1.0)
    discriminator_dfl = dfl.Constraints(0.0, {
        "rf": real_fake_dfl,
        "reg": discriminator_regularizer*generator_badness + (1.0 - generator_badness)
        # linear interpolation between the regularizer and 1.0 decided by the amount fooled
    })
    return generator_dfl, discriminator_dfl


def train_direct_step(env: ModelableEnv, generator, learning_rate):
    gen_optimizer = keras.optimizers.Adam(lr=learning_rate)

    @tf.function
    def gradient_step(batch):
        with tf.GradientTape() as gen_tape:
            generator_dfl = direct_dfls(generator, batch, env.closeness_dfl)
            generator_scalar = dfl.dfl_scalar(generator_dfl)
            generator_loss = 1-generator_scalar

            gen_grads = gen_tape.gradient(generator_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))

        return generator_scalar, generator_dfl

    def train_and_show(batch):
        return show_info(gradient_step(batch))

    return train_and_show


def train_GAN_step(env: ModelableEnv, generator, discriminator, learning_rate):
    gen_optimizer = keras.optimizers.Adam(lr=learning_rate)
    disc_optimizer = keras.optimizers.Adam(lr=learning_rate)

    @tf.function
    def gradient_step(batch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_dfl, discriminator_dfl = gan_dfls(generator, discriminator, batch, env.closeness_dfl)
            generator_scalar = dfl.dfl_scalar(generator_dfl)
            generator_loss = 1-generator_scalar
            discriminator_scalar = dfl.dfl_scalar(discriminator_dfl)
            discriminator_loss = 1-discriminator_scalar

        disc_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
        gen_grads = gen_tape.gradient(generator_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))

        return (generator_scalar, generator_dfl), (discriminator_scalar, discriminator_dfl)

    def train_and_show(batch):
        generator_info, discriminator_info = gradient_step(batch)
        return f"G: [{show_info(generator_info)}], D: [{show_info(discriminator_info)}]"

    return train_and_show


def show_info(scalar_constraints):
    scalar, constraints = scalar_constraints
    return f"{scalar:.2e}|{dfl.format_dfl(constraints)}"


def system_identify(env_name: str,
                    generator_hidden_sizes: list[int],
                    discriminator_hidden_sizes: list[int],
                    batch_size: int,
                    num_transitions: int,
                    num_batches: int,
                    epochs: int,
                    episode_size: int,
                    num_validation_batches: int,
                    save_freq: int,
                    learning_rate: float,
                    latent_size: int,
                    gan: bool,
                    load_saved: Union[str, None]):
    env = modelable_env.make_modelable(gym.make(env_name))  # make_modelable, create a loss for the env if it doesn't have one
    if load_saved:
        generator = tf.keras.models.load_model(load_saved)
    else:
        generator = generator_def(env, generator_hidden_sizes, latent_size)  # generator is a DNN for environment 
    discriminator = discriminator_def(env, discriminator_hidden_sizes, num_transitions)
    filepath = utils.random_subdir("models/" + env_name)
    dataset_spec = {
        "state": tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32)
        , "action": tf.TensorSpec(shape=env.action_space.shape, dtype=tf.float32)
        , "next_state": tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32)
        }

    dataset = tf.data.Dataset.from_generator(gather_mini_batch(env, episode_size), output_signature=dataset_spec) \
        .shuffle(episode_size*10).batch(num_transitions, drop_remainder=True).batch(batch_size, drop_remainder=True)

    validation_data = dataset.take(num_validation_batches).cache()
    pathlib.Path("caches").mkdir(parents=True, exist_ok=True)
    data = dataset.take(num_batches).apply(tf.data.experimental.assert_cardinality(num_batches)).cache(f"caches/{env_name}_nb:{num_batches}_e:{episode_size}_b:{batch_size}")
    if gan:
        trainer = train_GAN_step(env, generator, discriminator, learning_rate)
    else:
        trainer = train_direct_step(env, generator, learning_rate)

    def save_checkpoint(epoch):
        for batch in validation_data:
            print(show_info(direct_dfls(generator, batch, env.closeness_dfl)))
        utils.save_checkpoint(filepath, generator, epoch)

    utils.train_loop([data]*epochs, trainer, every_n_seconds={"freq": save_freq, "callback": save_checkpoint})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="gym environment", default="Pendulum-v2")
    parser.add_argument('--save_freq', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_transitions', type=int, default=1, help="number of transition tuples representing the amount of values a discriminator has to work with in a batch")
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--num_validation_batches', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gan', action='store_true', help="uses a GAN for learning a stochastic model of the transition function of the environment's observations")
    parser.add_argument('--load_saved', type=str, default=None)
    parser.add_argument('--generator_hidden_sizes', nargs="+", type=int, default=[100, 100])
    parser.add_argument('--discriminator_hidden_sizes', nargs="+", type=int, default=[300, 300])
    args = parser.parse_args()
    # tf.debugging.experimental.enable_dump_debug_info('my-tfdbg-dumps', tensor_debug_mode="FULL_HEALTH")
    system_identify(**vars(args))
