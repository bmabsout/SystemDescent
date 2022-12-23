import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from collections import namedtuple
import os
import pickle

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

def mlp(input_shape, hidden_sizes=(32,), activation='tanh', output_activation=None):

    model = tf.keras.Sequential()
    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))
    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation))
    model.build(input_shape=(None,) + input_shape)
    return model


def mlp_functional(inputs, hidden_sizes=(32,), activation='tanh', output_activation=None, use_bias=True):
    layer=inputs
    for hidden_size in hidden_sizes[:-1]:
        layer = tf.keras.layers.Dense(units=hidden_size, activation=activation)(layer)
    outputs = tf.keras.layers.Dense(
        units=hidden_sizes[-1],
        activation=output_activation,
        use_bias=use_bias,
        activity_regularizer=regularizers.l2(1e-4)
    )(layer)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

@tf.function
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class Mlp_Gaussian_Actor(tf.keras.Model):

  def __init__(self, obs_dim, ac_space, hidden_sizes, activation):
    super(Mlp_Gaussian_Actor, self).__init__()
    self.ac_space = ac_space
    self.obs_dim = obs_dim
    self.activation = activation
    self.hidden_sizes=hidden_sizes
    self.act_dim=ac_space.shape[-1]
    inputs = tf.keras.Input((obs_dim,))
    self.actor_mlp = mlp_functional(inputs, hidden_sizes=list(hidden_sizes)+[self.act_dim], activation=activation)
    deterministic_outputs = tf.keras.layers.Activation('sigmoid')(self.actor_mlp(inputs))*(self.ac_space.high - self.ac_space.low) + self.ac_space.low
    self.deterministic_actor = tf.keras.Model(inputs=inputs, outputs=deterministic_outputs)
    self.log_std = tf.clip_by_value(
        tf.Variable(name='log_std', shape=(self.act_dim,), initial_value=(-0.5,)*self.act_dim, trainable=True),
        LOG_STD_MIN, LOG_STD_MAX)

  def save(self, path):
    os.makedirs(path, exist_ok=True)
    pickle.dump((self.obs_dim, self.ac_space, self.hidden_sizes, self.activation), open(path+"/extra_data.pkl", "wb"))
    self.deterministic_actor.save(path+"/deterministic_actor")

  def load(path):
    actor = Mlp_Gaussian_Actor(*pickle.load(open(path+"/extra_data.pkl", "rb")))
    actor.deterministic_actor.set_weights(tf.keras.models.load_model(path+"/deterministic_actor").get_weights())
    return actor

  @tf.function
  def get_pi_logpi(self, observations):
    # tf.print(observations.shape)
    mu = self.actor_mlp(observations)
    std = tf.exp(self.log_std)
    pi = mu + tf.random.normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, self.log_std)
    logp_pi -= tf.reduce_sum((2*(tf.math.log(2.0) - pi - tf.math.softplus(-2*pi))),axis=1)
    pi_action = tf.math.sigmoid(pi)*(self.ac_space.high - self.ac_space.low) + self.ac_space.low

    return pi_action, logp_pi

  @tf.function
  def get_logp(self, observations, actions):
    mu = self.actor_mlp(observations)
    return gaussian_likelihood(actions, mu, self.log_std)


"""
Actor-Critics
"""
def mlp_actor_critic(obs_dim, ac_space, a_hidden_sizes=(32,32), q_hidden_sizes=(400,300), activation='relu'):
    act_dim = ac_space.shape[0]
    with tf.name_scope('pi'):
        actor_model = Mlp_Gaussian_Actor(obs_dim, ac_space, a_hidden_sizes, activation)
    with tf.name_scope('q1'):
        q1_network = mlp((obs_dim+act_dim,), list(q_hidden_sizes)+[1], activation, None)
    with tf.name_scope('q2'):
        q2_network = mlp((obs_dim+act_dim,), list(q_hidden_sizes)+[1], activation, None)
    ActorCritic = namedtuple('ActorCritic', 'pi q1 q2')
    return ActorCritic(actor_model, q1_network, q2_network)

