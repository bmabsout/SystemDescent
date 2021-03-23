import tensorflow as tf
import neuroflight_trainer.gyms
import gym
import os.path as osp
from pathlib import Path
import numpy as np
from gym.spaces import Box, Discrete
import argparse
import uuid
import rl_smoothness.envs.NNPendulum
import time
from tensorflow import keras
from tensorflow.keras import layers

def random_policy(obs, action_space):
    return action_space.sample()


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None, name=None):
    return tf.placeholder(dtype=tf.float32,name=name, shape=combined_shape(None, dim))


def placeholder_from_space(space, name=None):
    if isinstance(space, Box):
        return placeholder(space.shape, name)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32,name=name, shape=(None,))
    raise NotImplementedError


def mlp(input_layer, hidden_sizes=[32,32], activation=tf.nn.relu):
    layer=input_layer
    for h in hidden_sizes:
        layer = tf.layers.dense(layer, units=h, activation=activation)
    return layer


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


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


    shuffeled_indices = np.arange(len(prev_states))
    np.random.shuffle(shuffeled_indices)
    return {
        "prev_states": np.array(prev_states)[shuffeled_indices],
        "actions": np.array(actions)[shuffeled_indices],
        "states": np.array(states)[shuffeled_indices]
    }


def system_identify(env: gym.Env, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int):
    # with tf.scope
    with tf.variable_scope("dynamics"):
        action = placeholder_from_space(env.action_space, name="action")
        prev_state = placeholder_from_space(env.observation_space, name="prev_state")
        inputs = tf.concat([prev_state, action], axis=1)
        predicted_state = tf.layers.dense(mlp(inputs, hidden_sizes), units=prev_state.shape[1], name="predicted_state")
    
    state = placeholder_from_space(env.observation_space, name="state")

    cost = tf.reduce_mean((state - predicted_state)**2)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    tf.summary.scalar("cost", cost)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        uniq_id = uuid.uuid1().__str__()[:6]
        filepath = "saved/" + uniq_id
        summary_writer = tf.summary.FileWriter(osp.join(filepath, "tensorboard"), graph=tf.get_default_graph())
        for batch_index in range(0, mini_batches):
            if batch_index % save_freq:
                inputs = {"prev_state": prev_state, "action": action}
                outputs = {"predicted_state": predicted_state}
                path = osp.join(filepath, "checkpoints", "checkpoint{}".format(batch_index))
                print(path)
                tf.saved_model.simple_save(sess, export_dir=path, inputs=inputs, outputs=outputs),
                # pred = sess.run([predicted_state], feed_dict={ action: np.array([[0]]) ,prev_state: np.array([[0,1,2]])})
                # print("Predicted:", pred)
                # tf.io.write_graph(sess.graph, osp.join(filepath, "checkpoints"), "checkpoint{}.pbtxt".format(batch_index // save_freq))
            batch = gather_mini_batch(env, mini_batch_size, episode_size) #, policy=lambda o,a: np.array([0]))
            feed_dict= {
                prev_state:batch["prev_states"],
                action: batch["actions"],
                state: batch["states"]
            }
            val = sess.run([cost], feed_dict=feed_dict)
            print("Test error: ", val)
            for step in range(steps_per_batch):
                _, val, summary = sess.run([optimizer, cost, merged_summary_op],
                                        feed_dict=feed_dict)
                if step % 20 == 0:
                    print("batch: {}, step: {}, value: {}".format(batch_index+1, step, val))
                summary_writer.add_summary(summary, batch_index +step/steps_per_batch)
            


def system_identify_keras(env: gym.Env, hidden_sizes: list, mini_batches: int, mini_batch_size: int, steps_per_batch: int, episode_size: int, save_freq: int):
    km = keras_model(env, hidden_sizes)
    km.compile(
        optimizer=keras.optimizers.Adam(lr=0.001)
        , loss="mse")
    uniq_id = uuid.uuid1().__str__()[:6]
    filepath = "saved/" + uniq_id
        # summary_writer = tf.summary.FileWriter(osp.join(filepath, "tensorboard"), graph=tf.get_default_graph())
    for batch_index in range(0, mini_batches):
        batch = gather_mini_batch(env, mini_batch_size, episode_size) #, policy=lambda o,a: np.array([0]))
        km.fit(x=[batch["prev_states"], batch["actions"]], y=batch["states"], steps_per_epoch=steps_per_batch, batch_size=mini_batch_size)
        path = osp.join(filepath, "checkpoints")
        if batch_index % save_freq == 0:
            Path(path).mkdir(parents=True, exist_ok=True)
            filename = path+"/checkpoint{}.tf".format(batch_index)
            print("saving: ", filename)
            km.save(filename)
        # for i in range(steps_per_batch):
        #     err = km.train_on_batch(x=[batch["prev_states"], batch["actions"]], y=batch["states"])
        #     print(err)


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
    system_identify_keras(**vars(args))
    # gather_mini_batch(args.env, args.mini_batch_size)
