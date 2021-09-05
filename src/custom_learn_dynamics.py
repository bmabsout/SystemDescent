import tensorflow as tf
import gym
import os.path as osp
import numpy as np
from gym.spaces import Box, Discrete
import argparse
import time
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import utils
from typing import Union
import envs.Acrobot_continuous.Acrobot_continuous

def random_policy(obs, action_space):
    return action_space.sample()


def generator_def(env: gym.Env, hidden_sizes: list[int]):
    obs_space = env.observation_space
    act_space = env.action_space
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        print(act_space, obs_space)
        raise NotImplementedError
    state_size = obs_space.shape[0]
    state_input = keras.Input(shape=(obs_space.shape[0],))
    action_input = keras.Input(shape=(act_space.shape[0],))
    latent_input = keras.Input(shape=(1,))

    dense = layers.Concatenate()([state_input, action_input, latent_input])
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="selu",
                             kernel_initializer='lecun_normal',
                             kernel_regularizer=utils.PMean())(dense)
    low = np.array(obs_space.low)
    high = np.array(obs_space.high)

    dense = layers.Dense(state_size)(dense)
    outputs = layers.Activation('sigmoid')(dense)*(high-low) + low
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
        return tf.reshape(b, shape=tf.concat([[-1], b.shape[2:]], axis=0))

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
            if ep_len >= episode_size:
                obs = env.reset()
                prev_obs = obs
                ep_len = 0
    return true_generator


def discriminator_def(env: gym.Env, hidden_sizes: list[int], num_states: int):
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
def direct_dfls(generator, batch):
    abs_diff = tf.abs(generate_fakes(generator, batch)["next_state"]-batch["next_state"])
    return utils.p_mean(1.0/(1.0 + abs_diff), 1.0)


@tf.function
def gan_dfls(generator, discriminator, real_batch):
    fakes = generate_fakes(generator, real_batch)
    fakes2 = generate_fakes(generator, real_batch)
    fooled = utils.p_mean(discriminator(fakes), 2.0)
    discriminator_regularizer = 1 - tf.tanh(tf.reduce_mean(discriminator.losses)*10.0)
    generator_regularizer = 1 - tf.tanh(tf.reduce_mean(generator.losses)*10.0)
    real_fake_dfl = utils.DFL(0.0, {
        "real": utils.p_mean(discriminator(real_batch), 0.0),
        "fake": utils.p_mean(1 - discriminator(fakes), 0.0)
    })
    generator_dfl = utils.DFL(0.0, {
#         "fooled": utils.p_mean(discriminator(fakes2),2.0)
        "fooled": utils.smooth_constraint(utils.p_mean(discriminator(fakes2), 2.0), 0.0, 0.5, 0.0, 0.97, starts_linear=True),
        # "direct": direct_dfls(generator, real_batch)
        # "reg": tf.where(tf.stop_gradient(utils.dfl_scalar(real_fake_dfl)) < 0.1, generator_regularizer, 1.0)
    })
    
    discriminator_dfl = utils.DFL(0.0, {
        "rf": real_fake_dfl,
        "reg": tf.where(tf.stop_gradient(fooled) < 0.2, discriminator_regularizer, 1.0)
    })
    return generator_dfl, discriminator_dfl


# @tf.function
# def gan_dfls(generator, discriminator, real_batch):
#     fakes = generate_fakes(generator, real_batch)
#     generator_dfl = utils.DFL(2.0, {
#         "fooled": utils.p_mean(discriminator(fakes), 1.0,slack=1e-5),
#         # "direct": direct_dfls(generator, real_batch)
#     })
#     discriminator_dfl = utils.DFL(0.0, {
#         "real": utils.p_mean(discriminator(real_batch), 0.0),
#         "fake": utils.p_mean(1 - discriminator(fakes), 0.0, slack=1e-10)
#     })
#     return generator_dfl, discriminator_dfl


def train_direct_step(generator, learning_rate):
    gen_optimizer = keras.optimizers.Adam(lr=learning_rate)

    @tf.function
    def gradient_step(batch):
        for i in range(1):
            with tf.GradientTape() as gen_tape:
                generator_dfl = direct_dfls(generator, batch)
                generator_scalar = utils.dfl_scalar(generator_dfl)
                generator_loss = 1-generator_scalar

            gen_grads = gen_tape.gradient(generator_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))

        return generator_scalar, generator_dfl

    def train_and_show(batch):
        return show_info(gradient_step(batch))

    return train_and_show


train_discriminator = True


def train_GAN_step(generator, discriminator, learning_rate):
    global train_discriminator
    gen_optimizer = keras.optimizers.Adam(lr=learning_rate)
    disc_optimizer = keras.optimizers.Adam(lr=3e-4)

    @tf.function
    def gradient_step(batch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_dfl, discriminator_dfl = gan_dfls(generator, discriminator, batch)
            generator_scalar = utils.dfl_scalar(generator_dfl)
            generator_loss = 1-generator_scalar
            discriminator_scalar = utils.dfl_scalar(discriminator_dfl)
            discriminator_loss = 1-discriminator_scalar

        disc_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
        gen_grads = gen_tape.gradient(generator_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))
    # else:
        # tf.print(utils.mean_grad_size(gen_grads))
        # tf.print(utils.mean_grad_size(disc_grads))
        # tf.print(utils.mean_grad_size(discriminator.trainable_weights))
        return (generator_scalar, generator_dfl), (discriminator_scalar, discriminator_dfl)

    def train_and_show(batch):
        generator_info, discriminator_info = gradient_step(batch)
        return f"G: [{show_info(generator_info)}], D: [{show_info(discriminator_info)}]"

    return train_and_show


def show_info(scalar_dfl):
    scalar, dfl = scalar_dfl
    return f"{scalar:.2e}|{utils.format_dfl(dfl)}"


def system_identify(env_name: str,
                    generator_hidden_sizes: list[int],
                    discriminator_hidden_sizes: list[int],
                    batch_size: int,
                    num_states: int,
                    num_batches: int,
                    epochs: int,
                    episode_size: int,
                    num_validation_batches: int,
                    save_freq: int,
                    learning_rate: float,
                    load_saved: Union[str, None]):
    env = gym.make(env_name)
    if load_saved:
        generator = tf.keras.models.load_model(load_saved)
    else:
        generator = generator_def(env, generator_hidden_sizes)
    discriminator = discriminator_def(env, discriminator_hidden_sizes, num_states)
    filepath = utils.random_subdir("models/" + env_name)
    dataset_spec = ({
        "state": tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32)
        , "action": tf.TensorSpec(shape=env.action_space.shape, dtype=tf.float32)
        , "next_state": tf.TensorSpec(shape=env.observation_space.shape, dtype=tf.float32)
        })

    dataset = tf.data.Dataset.from_generator(gather_mini_batch(env, episode_size), output_signature=dataset_spec) \
        .shuffle(episode_size*10).batch(num_states, drop_remainder=True).batch(batch_size, drop_remainder=True)

    validation_data = dataset.take(num_validation_batches).cache()
    pathlib.Path("caches").mkdir(parents=True, exist_ok=True)
    data = dataset.take(num_batches).apply(tf.data.experimental.assert_cardinality(num_batches)).cache(f"caches/{env_name}_nb:{num_batches}_e:{episode_size}_b:{batch_size}")
    trainer = train_GAN_step(generator, discriminator, learning_rate)
    # trainer = train_direct_step(generator, learning_rate)

    def end_of_epoch(epoch, dt):
        global train_discriminator
        train_discriminator = not train_discriminator
        utils.save_checkpoint(filepath, generator, epoch)

    utils.train_loop([data]*epochs, trainer, end_of_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="gym environment", default="Pendulum-v0")
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_states', type=int, default=1)
    parser.add_argument('--num_batches', type=int, default=10000)
    parser.add_argument('--num_validation_batches', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--load_saved', type=str, default=None)
    parser.add_argument('--generator_hidden_sizes', nargs="+", type=int, default=[200,100])
    parser.add_argument('--discriminator_hidden_sizes', nargs="+", type=int, default=[200, 100])
    args = parser.parse_args()
    # tf.debugging.experimental.enable_dump_debug_info('my-tfdbg-dumps', tensor_debug_mode="FULL_HEALTH")
    system_identify(**vars(args))
