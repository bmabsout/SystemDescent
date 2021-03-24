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


def compile_lyapunov_model(X, state_shape, epsilon=0.001, hyper0=1, hyper_diff=1):
	# X: The dataset 
	# env: a gym env

	def V_def():
		inputs = keras.Input(shape=state_shape)
		dense1 = layers.Dense(64, activation='relu')(inputs)
		dense2 = layers.Dense(64, activation='relu')(dense1)
		outputs = layers.Dense(1)(dense2)
		model = keras.Model(inputs=inputs, outputs=outputs, name="V")
		print()
		print()
		print("V:")
		model.summary()
		return model

	def full_model_def(V):
		input_x=keras.Input(shape=state_shape)
		input_fxu=keras.Input(shape=state_shape)
		input_0=keras.Input(shape=state_shape)

		model = keras.Model(inputs=[input_x,input_fxu,input_0], outputs=[V(input_x),V(input_fxu),V(input_0)], name="full_model")
		print()
		print()
		print("full_model:")
		model.summary()
		return model


	def lyapunov_loss(_, y_pred):
		# V((1,0,0)) = 0 for no movement and pointing upwards
		# V(f(x,u)) - V(x) < 0
		V_x = y_pred[0]
		V_fxu = y_pred[1]
		V_0 = y_pred[2]
		return hyper0*tf.reduce_mean(V_0**2) + hyper_diff*tf.reduce_mean(tf.nn.relu(epsilon + V_x - V_fxu))


	full_model = full_model_def(V_def())
	full_model.compile(loss=lyapunov_loss, optimizer=keras.optimizers.Adam(lr=0.01))
	return full_model

def generate_dataset(dynamics_model, num_samples):
	x = np.array([env.observation_space.sample() for _ in range(num_samples)])
	u = np.array([env.action_space.sample() for _ in range(num_samples)])
	fxu = dynamics_model.predict([x,u])
	p = np.repeat([[1,0,0]], num_samples, axis=0) 
	# (1,0,0) means no movement and pointing upwards
	return [x, fxu, p]

env = gym.make('Pendulum-v0')
dynamics_model = keras.models.load_model("/home/bmabsout/Documents/gymfc-nf1/training_code/neuroflight_trainer/dynamics_learning/saved/ff44f9/checkpoints/checkpoint50.tf")
print()
print()
print("dynamics_model:")
dynamics_model.summary()

X = generate_dataset(dynamics_model,100_000)
lyapunov_model = compile_lyapunov_model(X, env.observation_space.shape)
lyapunov_model.fit(X, epochs=10, batch_size=1024)