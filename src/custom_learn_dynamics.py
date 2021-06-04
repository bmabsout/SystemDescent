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

    @tf.function
    def repeat_train(n, batch):
        maxRepetitions = 15
        repetitions = tf.random.uniform(shape=[], minval=7, maxval=maxRepetitions+1, dtype=tf.dtypes.int32)
        for i in range(n):
            loss_value, metrics = train_step(batch, repetitions, maxRepetitions)
        return loss_value, metrics

def train_loop(epochs, batches):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        for step, batch in enumerate(batches):
            
            loss_value, metrics = repeat_train(5, batch)
            # Log every 200 batches.
            if step % 2 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4e, %s"
                    % (step, float(loss_value), str(map_dict_elems(lambda v: f"{v:.4e}", metrics)))
                )
                print("Seen so far: %d samples" % ((step + 1) * args.batch_size))

        save_model(actor, "actor_tf")
        save_model(lyapunov_model, "lyapunov_tf")
        print("Time taken: %.2fs" % (time.time() - start_time))


def system_identify(env_name: str, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int, learning_rate: float):
    env = gym.make(env_name)
    model = model_def(env, hidden_sizes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate)
        , loss="mse")
    filepath = utils.random_subdir("models/" + env_name)
    for batch_index in range(1, mini_batches+1):
        batch = gather_mini_batch(env, mini_batch_size, episode_size)
        test_batch = gather_mini_batch(env, mini_batch_size//2, episode_size)
        model.fit(x=[batch["prev_states"], batch["actions"]], y=batch["states"], epochs=steps_per_batch, validation_data=([test_batch["prev_states"], test_batch["actions"]], test_batch["states"]), batch_size=2048)
        if batch_index % save_freq == 0:
            utils.save_checkpoint(filepath, model, batch_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="gym environment", default="Pendulum-v0")
    parser.add_argument('--mini_batches', type=int, default=400)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--episode_size', type=int, default=200)
    parser.add_argument('--mini_batch_size', type=int, default=100000)
    parser.add_argument('--steps_per_batch', type=int, default=100)
    parser.add_argument('--hidden_sizes', nargs="+", type=int, default=[256,256])
    args = parser.parse_args()
    system_identify(**vars(args))
