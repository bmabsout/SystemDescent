import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from typing import Tuple
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import losses
from functools import reduce
from pathlib import Path
import time
import argparse
from utils import *
from tqdm import tqdm
import pprint

def V_def(state_shape: Tuple[int, ...]):
	input_state = keras.Input(shape=state_shape)
	input_setpoint = keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_setpoint])
	dense1 = layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
	dense2 = layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
	outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
	model = keras.Model(inputs={"state": input_state, "setpoint": input_setpoint}, outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	input_state = keras.Input(shape=state_shape)
	input_set_point = keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_set_point])
	dense1 = layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
	dense2 = layers.Dense(32, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
	# dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	prescaled = layers.Dense(np.squeeze(action_shape), activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
	outputs = prescaled*2.0
	model = keras.Model(inputs={"state": input_state, "setpoint": input_set_point}, outputs=outputs)
	model.summary()
	return model

def generate_dataset(env):
	def gen_sample():
		while True:
			obs = env.reset()
			obs[2] = obs[2]*7.0
			init_state = np.random.uniform(-np.pi, np.pi)
			angle = np.where(np.abs(np.cos(init_state)) < 0.7, 0.0, init_state)
			# chooses only non-sideways angles

			# return [obs, [np.cos(angle), np.sin(angle), 0.0]]
			yield {"state": obs, "setpoint":[1.0, 0.0, 0.0]}
	return gen_sample

def save_model(model, name):
	path = Path(args.ckpt_path, name)
	path.mkdir(parents=True, exist_ok=True)
	print(str(path))
	model.save(str(path))

def train(batches, dynamics_model, actor, V, state_shape, args):
	optimizer=keras.optimizers.Adam(lr=args.lr)
	@tf.function
	def run_full_model(initial_states, set_points, repeat=1):
		# tf.print(initial_states)
		states = tf.TensorArray(tf.float32, size=repeat)
		current_states = initial_states
		for i in range(repeat):
			current_states = dynamics_model(
				{ "state": current_states
				, "action": actor({"state":current_states, "setpoint":set_points})
				}, training=True)
			states = states.write(i, current_states)
		return current_states, tf.transpose(states.stack(), [1,0,2])

	@tf.function
	def batch_value(batch):
		maxRepetitions = 7
		repetitions = tf.random.uniform(shape=[], minval=5, maxval=maxRepetitions+1, dtype=tf.dtypes.int32)
		initial_states = batch["state"]
		set_points = batch["setpoint"]
		fxu, states = run_full_model(initial_states, set_points,repeat=repetitions)
		Vx = V({"state": initial_states, "setpoint": set_points}, training=True)
		V_fxu = V({"state": fxu, "setpoint": set_points}, training=True)
		zero = p_mean(1.0-V({"state": set_points, "setpoint": set_points}), 0)**2.0
		diff = (Vx - V_fxu)
		transposed_states = tf.transpose(states, [2,0,1])
		transposed_setpoints = tf.broadcast_to(tf.expand_dims(tf.transpose(set_points), axis=-1), tf.shape(transposed_states))
		as_all = angular_similarity(transposed_states, transposed_setpoints)
		close_angle = p_mean(angular_similarity(tf.transpose(fxu)[0:2] ,tf.transpose(set_points)[0:2]), 1.)

		actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
		lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

		avg_large = tf.minimum(transform(p_mean(Vx,-1.0, slack=1e-15), 0.0, 0.4, 0.0, 1.0), 1.0)
		repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
		maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
		line = tf.math.tanh(transform(repetitionsf, 0.05, 4.0*maxRepetitionsf*Vx**0.5+0.05, 0.0, 1.0))*Vx**0.5
		decreases_everywhere = smooth_constraint(diff, 0.0, line)
		proof_of_performance = tf.squeeze(p_mean(decreases_everywhere, -1.0, slack=1e-13))

		dfl = DFL(1.0,
			{
			# "close_angles": p_mean(as_all, 3.0),
			"close_angle": tf.clip_by_value(transform(close_angle,0.4, 0.6, 0.0, 1.0), 0.0, 1.0),
			"lyapunov": DFL(0.0, {
				"proof_of_performance": proof_of_performance,
				# "avg_large": tf.minimum(p_mean(Vx, -2.0)*1.5, 1.0),
				"zero": zero,
				"actor_reg": tf.minimum(transform(actor_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
				"lyapunov_reg": tf.minimum(transform(lyapunov_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
			})
		})

		return dfl_scalar(dfl), dfl

	@tf.function
	def train_step(batch):
		for i in tf.range(1):
			with tf.GradientTape() as tape:
				scalar, metrics = batch_value(batch)
				# loss_value = scale_gradient(loss_value, 1/loss_value**4.0)
				loss = 1 - scalar
				# tf.print(value)
			grads = tape.gradient(loss, actor.trainable_weights + V.trainable_weights)
			# tf.print(grads)
			# tf.print(sum(map(lambda grad_bundle: tf.reduce_mean(tf.abs(grad_bundle)), grads))/len(grads))
			# tf.print(1-loss_value)
			optimizer.apply_gradients(zip(grads, actor.trainable_weights + V.trainable_weights))

		return scalar, metrics

	def save_models():
		save_model(actor, "actor_tf")
		save_model(lyapunov_model, "lyapunov_tf")

	def train_and_show(batch):
		scalar, metrics = train_step(batch)
		return f"Scalar: {scalar:.2e}|||{format_dfl(metrics)}"

	train_loop([batches]*args.epochs, train_and_show, save_models)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_path",type=str,default=latest_model())
	parser.add_argument("--num_batches",type=int,default=200)
	parser.add_argument("--epochs",type=int,default=100)
	parser.add_argument("--batch_size", type=int,default=256)
	parser.add_argument("--lr",type=float, default=1e-3)
	parser.add_argument("--load_saved", action="store_true")
	args = parser.parse_args()

	env_name = extract_env_name(args.ckpt_path)
	env = gym.make(env_name)

	action_shape = env.action_space.shape
	state_shape = env.observation_space.shape

	dynamics_model = keras.models.load_model(args.ckpt_path)
	print()
	print()
	print("dynamics_model:")
	dynamics_model.summary()
	actor = (
			keras.models.load_model(args.ckpt_path + "/actor_tf")
		if args.load_saved else
			actor_def(state_shape, action_shape)
	)

	lyapunov_model = (
			keras.models.load_model(args.ckpt_path + "/lyapunov_tf")
		if args.load_saved else
			V_def(state_shape)
	)
	state_spec = tf.TensorSpec(state_shape)
	dataset_spec = { "state": state_spec , "setpoint": state_spec}
	dataset = tf.data.Dataset.from_generator(generate_dataset(env), output_signature=dataset_spec)
	batched_dataset = dataset.batch(args.batch_size).take(args.num_batches).cache()
	train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)

