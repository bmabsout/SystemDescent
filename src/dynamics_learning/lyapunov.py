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
from functools import reduce
from pathlib import Path
import argparse

def V_def(state_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(64, activation='sigmoid')(inputs)
	dense2 = layers.Dense(64, activation='sigmoid')(dense1)
	outputs = layers.Dense(1, activation='sigmoid')(dense2)
	model = keras.Model(inputs=inputs, outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(4, activation='sigmoid')(inputs)
	# dense2 = layers.Dense(64, activation='sigmoid')(dense1)
	outputs = layers.Dense(np.squeeze(action_shape), activation='sigmoid')(dense1)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model

def friction_actor_def():
	inputs=keras.Input(shape=(3,))
	outputs = layers.Lambda(lambda x: -0.9*x[:,2])(inputs)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model

def tf_geo_mean(l, **kwargs):
	return tf.exp(tf.reduce_mean(tf.math.log(l+1e-10),**kwargs))-1e-10

def full_model_def(f, actor, V, state_shape, args):
	input_x=keras.Input(shape=state_shape)
	# input_fxu=keras.Input(shape=state_shape)
	action = actor(input_x)
	f.training=False
	fxu = f([input_x, action])
	input_0=keras.Input(shape=state_shape)
	V0 = tf.squeeze(V(input_0))
	Vx = tf.squeeze(V(input_x))
	V_fxu = tf.squeeze(V(fxu))
	# loss = layers.Lambda(lambda x: x)([loss_V0])

	model = keras.Model(inputs=[input_x,input_0], outputs=[V0, Vx, V_fxu], name="full_model")
	
	loss_V0 = tf.reduce_mean(1-V0,keepdims=True)
	loss_Vx = tf.reduce_mean(Vx, keepdims=True)
	loss_V_fxu = tf_geo_mean((Vx - V_fxu)/2. + 0.5, keepdims=True)*(tf_geo_mean((Vx - V_fxu)**2, keepdims=True))
	final_loss = 1-geo_mean([loss_Vx, loss_V0, loss_V_fxu])
	# loss_V0 = tf.reduce_mean(V0**2, keepdims=True)
	# loss_Vx = tf.reduce_mean(tf.nn.relu(args.epsilon_x - Vx), keepdims=True)
	# loss_V_fxu = tf.reduce_mean(tf.nn.relu(args.epsilon_diff + Vx - V_fxu ))
	# final_loss = loss_V0 + args.hyper_psd*loss_Vx + args.hyper_diff*loss_V_fxu

	model.add_loss(final_loss)
	model.add_metric(loss_V0, name="zero", aggregation="mean")
	model.add_metric(loss_Vx, name="large", aggregation="mean")
	model.add_metric(loss_V_fxu, name="decreasing", aggregation="mean")
	print()
	print()
	model.summary()
	return model

def geo_mean(l):
	return reduce(lambda a,b: a*b,map(lambda e: (e+1e-10)**(1/len(l)),l)) - 1e-10

def generate_dataset(dynamics_model, num_samples):
	x = np.array([env.observation_space.sample() for _ in range(num_samples)])
	# u = np.array([env.action_space.sample() for _ in range(num_samples)])
	# fxu = dynamics_model.predict([x,u])
	p = np.repeat([[-1,0,0]], num_samples, axis=0) 
	# (1,0,0) means no movement and pointing upwards
	return [x, p]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="./saved/f6bfa5/checkpoints/checkpoint4")
parser.add_argument("--num_samples",type=int,default=100_000)
parser.add_argument("--epochs",type=int,default=5)
parser.add_argument("--batch_size", type=int,default=10024)
parser.add_argument("--epsilon_x", type=float,default=0.3)
parser.add_argument("--epsilon_diff", type=float,default=1)
parser.add_argument("--lr",type=float, default=1e-3)
parser.add_argument("--hyper_diff", type=float, default=10.0)
parser.add_argument("--hyper_psd", type=float, default=0.1)

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
# actor = friction_actor_def()
lyapunov_model = V_def(state_shape)
full_model = full_model_def(dynamics_model, actor, lyapunov_model, state_shape, args)
full_model.compile(loss=[None]*len(full_model.outputs), optimizer=keras.optimizers.Adam(lr=args.lr))

for i in range(args.epochs):
	X = generate_dataset(dynamics_model,args.num_samples)
	full_model.fit(X, epochs=10, batch_size=args.batch_size)
	# print()

def save_model(model, name):
	path = Path(args.ckpt_path, name)
	path.mkdir(parents=True, exist_ok=True)
	print(str(path))
	model.save(str(path))


save_model(actor, "actor_tf")
save_model(lyapunov_model, "lyapunov_tf")
