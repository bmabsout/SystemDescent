import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import losses
import argparse

def V_def(state_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(1023, activation='relu')(inputs)
	dense2 = layers.Dense(1023, activation='relu')(dense1)
	outputs = layers.Dense(1)(dense2)
	model = keras.Model(inputs=inputs, outputs=outputs, name="V")
	print()
	print()
	print("V:")
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(64)(inputs)
	outputs = layers.Dense(np.squeeze(action_shape))(dense1)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model

def full_model_def(f, actor, V, state_shape):
	input_x=keras.Input(shape=state_shape)
	# input_fxu=keras.Input(shape=state_shape)
	action = actor(input_x)
	fxu = f([input_x, action])
	input_0=keras.Input(shape=state_shape)

	model = keras.Model(inputs=[input_x,input_0], outputs=[V(input_x),V(fxu),V(input_0)], name="full_model")
	print()
	print()
	print("full_model:")
	model.summary()
	return model


def lyapunov_loss(_, y_pred):
	# V((1,0,0)) = 0 for no movement and pointing upwards
	# V(f(x,u)) - V(x) < 0
            # V(x) > 0
	V_x = y_pred[0]
	V_fxu = y_pred[1]
	V_0 = y_pred[2]
	return tf.reduce_mean(V_0**2) + args.hyper_diff*tf.reduce_mean(tf.nn.relu(args.epsilon + V_fxu - V_x )) + args.hyper_psd * tf.reduce_mean(tf.nn.relu(args.epsilon*10 - V_x))

def generate_dataset(dynamics_model, num_samples):
	x = np.array([env.observation_space.sample() for _ in range(num_samples)])
	# u = np.array([env.action_space.sample() for _ in range(num_samples)])
	# fxu = dynamics_model.predict([x,u])
	p = np.repeat([[1,0,0]], num_samples, axis=0) 
	# (1,0,0) means no movement and pointing upwards
	return [x, p]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="./saved/428b23/checkpoints/checkpoint210.tf")
parser.add_argument("--num_samples",type=int,default=1000_000)
parser.add_argument("--epochs",type=int,default=5)
parser.add_argument("--batch_size", type=int,default=2024)
parser.add_argument("--epsilon", type=float,default=0.01)
parser.add_argument("--lr",type=float, default=1e-2)
parser.add_argument("--hyper_diff", type=float, default=1.0)
parser.add_argument("--hyper_psd", type=float, default=1.0)

args = parser.parse_args()

env = gym.make('Pendulum-v0')
print(env.action_space.shape)
action_shape = env.action_space.shape
state_shape = env.observation_space.shape
dynamics_model = keras.models.load_model(args.ckpt_path)
print()
print()
print("dynamics_model:")
dynamics_model.summary()

actor = actor_def(state_shape, action_shape)
lyapunov_model = V_def(state_shape)

full_model = full_model_def(dynamics_model, actor, lyapunov_model, state_shape)
full_model.compile(loss=lyapunov_loss, optimizer=keras.optimizers.Adam(lr=args.lr))

X = generate_dataset(dynamics_model,args.num_samples)
full_model.fit(X, epochs=args.epochs, batch_size=args.batch_size)

lyapunov_model.save("lyapunov.tf")

actor.save(args.ckpt_path +".actor.tf")