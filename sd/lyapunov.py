import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from typing import Tuple
from os import path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras import losses
from functools import reduce
from pathlib import Path
import argparse
from .dfl import *
from . import utils
from tqdm import tqdm
import sd.envs

def V_def(state_shape: Tuple[int, ...], input_setpoint_shape=None):
	input_state = keras.Input(shape=state_shape)
	# input_setpoint = keras.Input(shape=state_shape)
	input_setpoint = keras.Input(shape=input_setpoint_shape) if input_setpoint_shape else keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_setpoint])
	dense1 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
	dense2 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
	outputs = layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
	# outputs = layers.Activation("relu")(activation)

	# outputs = layers.Lambda(lambda x: tf.clip_by_value(x+0.5, 0.0, 1.0))(activation)
	model = keras.Model(inputs={"state": input_state, "setpoint": input_setpoint}, outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def actor_def(state_shape, action_shape, input_setpoint_shape=None):
	input_state = keras.Input(shape=state_shape)
	input_set_point = keras.Input(shape=input_setpoint_shape) if input_setpoint_shape else keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_set_point])
	dense1 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
	dense2 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
	# dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	prescaled = layers.Dense(np.squeeze(action_shape), activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense2)
	outputs = prescaled*2.0
	model = keras.Model(inputs={"state": input_state, "setpoint": input_set_point}, outputs=outputs)
	model.summary()
	return model

def generate_dataset(env: gym.Env):
	def gen_sample():
		"""Generates a sample from the environment but assumes that the environment is a pendulum"""
		while True:
			obs, _ = env.reset()

			## PENDULUM START
			# obs[2] = obs[2]*7.0
			# # angle = np.where(np.abs(np.cos(init_state)) < 0.7, 0.0, init_state) # randomize setpoints
			# # angle = np.random.uniform(-np.pi/7.0, np.pi/7.0) + np.random.randint(2)*np.pi
			# # yield {"state":obs, "setpoint": [np.cos(angle), np.sin(angle), 0.0]}
			# yield {"state": obs, "setpoint":[1.0, 0.0, 0.0]} 
			# # yield {"state": obs, "setpoint":[1.0, 0.0, 0.0]} if np.random.randint(2) == 1 else {"state": obs, "setpoint":[-1.0,0.0,0.0]}
			## PENDULUM END

			## AMAZINGBALL START
			yield {"state": np.array(obs), "setpoint": np.array([0.0, 0.0, 0.0, 0.0])}
			## AMAZINGBALL END
	return gen_sample

def save_model(model, name):
	path = Path(args.ckpt_path.parent, name)
	args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
	
	print(str(path))
	model.save(str(path))

pi = tf.constant(math.pi)

# @tf.function
def angular_similarity(v1, v2):
    v1_angle = tf.math.atan2(v1[1], v1[0]) # atan2's range [-pi, pi]
    v2_angle = tf.math.atan2(v2[1], v2[0])
    return tf.abs(tf.abs(v1_angle - v2_angle) - pi) / pi

def abs_base_loss_signal(transposed_states, transposed_setpoints):
	"""Calculate loss from system state and setpoint, without any Lyapunov parts
	
	Args:
		transposed_states shape   : (state_dim:8, batch_size, repeat)
		transposed_setpoints shape: (state_dim:4, batch_size, repeat)

	Returns:
		loss value
	"""
	raise NotImplementedError("abs_base_loss_signal is not implemented. Use end state loss.")

def abs_fxu_loss(final_states, setpoints):
	"""Calculate loss from ending states and setpoints, without any Lyapunov parts
	
	Args:
		final_states shape: (batch_size, state_dim:8)
		setpoints shape   : (batch_size, state_dim:4)
		
	Returns:
		loss tensor scalar
	"""
	raise NotImplementedError("abs_fxu_loss is not implemented")
	


def train(batches, dynamics_model, actor, V, state_shape, args):
	# optimizer=keras.optimizers.Adam(lr=args.lr)
	optimizer=keras.optimizers.Adam(learning_rate=args.lr)
	# @tf.function
	def run_full_model(initial_states, set_points, repeat=1):
		'''Runs the dynamics model for repeat steps and returns the final state and the states at each step'''
		states = tf.TensorArray(tf.float32, size=repeat)
		current_states = initial_states
		batch_size = tf.shape(initial_states)[0]
		latent_shape = (batch_size,) + tuple(dynamics_model.input["latent"].shape[1:])
		for i in range(repeat):
			current_states = dynamics_model(
				{ "state": current_states
				, "action": actor({"state":current_states, "setpoint":set_points})
				, "latent": tf.random.normal(latent_shape)
				}, training=True)  
			states = states.write(i, current_states)
		return current_states, tf.transpose(states.stack(), [1,0,2]) # ok, I think this operation is to put batch back to the first dimension

	# @tf.function
	def batch_value(batch):
		'''
		   batch: {
			   "state": [batch_size, state_dim]
			   "setpoint": [batch_size, state_dim]
		   }

		   state_dim: 3: [cos(theta), sin(theta), theta_dot]

		'''
		maxRepetitions = 15
		#repetitions = tf.random.uniform(shape=[], minval=10, maxval=maxRepetitions+1, dtype=tf.dtypes.int32)
		repetitions = tf.random.uniform(shape=[], minval=1, maxval=maxRepetitions+1, dtype=tf.dtypes.int32) # whether small minval can shrink the local minimal
		prev_states = batch["state"]
		set_points = batch["setpoint"]
		set_points_dim = tf.shape(set_points)[-1]

		# repetitions is a random int, fxu is the final state after that many steps updates.
		# states are a collection of states at each step
		fxu, states = run_full_model(prev_states, set_points,repeat=repetitions)
		Vx = V({"state": prev_states, "setpoint": set_points}, training=True) 
		
		# the Lyapunov value at the final state after the update steps
		V_fxu = V({"state": fxu, "setpoint": set_points}, training=True)

		# the Lyapunov value at the setpoint(origin) should be zero
		# thus a fully trained V(setpoint) should return zero. Thus zero == 1 when sufficiently trained
		if fxu.shape[1] != set_points.shape[1]:
			zero_states = tf.concat([fxu[:,:set_points_dim], set_points], axis=1)
		else:
			zero_states = set_points

		zero = p_mean((1.0-V({"state": zero_states, "setpoint": set_points})**0.5), -1.0)
		
		# for condition: V shall decrease along time (i.e. along the update steps)
		# diff = (Vx - V_fxu)

		diff = (Vx - V_fxu)

		transposed_states = tf.transpose(states, [2,0,1])
		tmp_ts = tf.expand_dims(tf.transpose(set_points), axis=-1)
		target_shape = [tmp_ts.shape[0], tmp_ts.shape[1], transposed_states.shape[2]] if transposed_states.shape[0] != tmp_ts.shape[0] else transposed_states.shape
		# transposed_setpoints = tf.broadcast_to(tmp_ts, tf.shape(transposed_states))
		transposed_setpoints = tf.broadcast_to(tmp_ts, target_shape)
		
		as_all = angular_similarity(transposed_states, transposed_setpoints)
		angular_similarities = angular_similarity(tf.transpose(fxu)[0:2] ,tf.transpose(set_points)[0:2])

		actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
		lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

		# if near the setpoint, decrease slower. otherwise decrease faster. This shapes the Lyapunov function. 
		repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
		maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
		decrease_by = 1.0/100.0 # should arrive to the target within 100 steps, think about maximizing this parameter
		line = tf.minimum(decrease_by*repetitionsf, Vx) # how much we would like taking a step to reduce V by
		proof_of_performance = p_mean( build_piecewise([(-1.0, 0.0), (-0.1, 1e-5), (0.0, 0.01), (line, 0.9), (1.0, 1.0)], diff, clipped=True), -2.0)
		# for now proof of performance has a hardcoded piecewise linear function for the ranges that we consider critical (negative values) vs nice to have (above line)
			
		non_setpoint_Vx = tf.where(angular_similarities > 0.99, 1.0, Vx)
		large_elsewhere = p_mean(tf.minimum(non_setpoint_Vx*2, 1.0), -2.0) # making sure non setpoints Vx > 0.1

		dfl = Constraints(0.0,
			{
			# "close_angles": scale_gradient(p_mean(as_all, 2.0), 1.0),
			# "close_angles": build_piecewise([(0.0, 0.0), (0.6, 0.01), (0.7, 0.9), (1.0, 1.0)], p_mean(as_all, 2.0)),
			"lyapunov": Constraints(0.0, {
				"pop": proof_of_performance,
				"large": large_elsewhere,
				"zero": zero,
			# 	# "actor_reg": tf.minimum(transform(actor_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
			# 	# "lyapunov_reg": tf.minimum(transform(lyapunov_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
			})
		})

		return dfl

	# @tf.function
	def set_gradient_size(gradients, size):
		return size*gradients/tf.norm(gradients)

	# @tf.function
	def train_step(batch):
		# for i in tf.range(1):
		with tf.GradientTape() as tape:
			dfl = batch_value(batch)
			# loss_value = scale_gradient(loss_value, 1/loss_value**4.0)

			# the scalar is best to be 1, so that the loss is best to be 0.
			scalar = dfl_scalar(dfl)
			loss = 1-scalar
			# tf.print(value)
		grads = tape.gradient(loss, actor.trainable_weights + V.trainable_weights)
		# modified_grads = [ (grad_bundle if grad_bundle is None else set_gradient_size(grad_bundle, loss)) for grad_bundle in grads ]
		# tf.print(grads)
		# optimizer.learning_rate = args.lr*transform(scalar, 0.3, 0.7, 1.0, 0.01, clipped=True)
		# tf.print(1-loss_value)
		# tf.print("grads:", grads)
		# tf.print("trainable:", V.trainable_weights)
		# tf.print(tf.reduce_mean(list(map(lambda x: tf.reduce_mean(tf.abs(x)), grads))))
		# mean_grad_size = tf.reduce_mean([ tf.reduce_mean(tf.abs(tensor)) for tensor in grads[0:len(actor.trainable_weights)]])
		# tf.print(mean_grad_size)
		optimizer.apply_gradients(zip(grads, actor.trainable_weights + V.trainable_weights))

		return scalar, dfl

	def save_models(epoch):
		save_model(actor, "actor.keras")
		save_model(lyapunov_model, "lyapunov.keras")

	def train_and_show(batch):
		scalar, metrics = train_step(batch)
		return f"Scalar: {scalar:.2e}|||{metrics}"

	utils.train_loop([batches]*args.epochs,
		train_step=train_and_show,
		every_n_seconds={"freq": args.save_freq, "callback": save_models})
	
if __name__ == "__main__":
	# tf.config.run_functions_eagerly(True)
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_path",type=Path,default=None)
	parser.add_argument("--num_batches",type=int,default=200)
	parser.add_argument("--save_freq",type=int,default=15, help="save the checkpoints every n seconds")
	parser.add_argument("--epochs",type=int,default=100)
	parser.add_argument("--batch_size", type=int,default=128)
	parser.add_argument("--lr",type=float, default=1e-2)
	parser.add_argument("--load_saved", action="store_true")
	args = parser.parse_args()
	if args.ckpt_path is None:
		args.ckpt_path = utils.latest_model()

	env_name = utils.extract_env_name(args.ckpt_path)
	print("env_name:", env_name)
	env = gym.make(env_name)

	# action_shape = env.action_space.shape
	# state_shape = env.observation_space.shape
	action_shape = utils.infer_shape(env, 'action_space')
	state_shape = utils.infer_shape(env, 'observation_space')
	try:
		setpoint_shape = utils.infer_shape(env, 'setpoint_space')
	except AssertionError:
		print('setpoint_space not specified in env, using observation_space instead')
		setpoint_shape = state_shape

	dynamics_model = utils.load_checkpoint(args.ckpt_path)
	print()
	print()
	print("dynamics_model:")
	dynamics_model.summary()
	
	# if --load_saved is not set, actor_def and V_def are used to define the actor and lyapunov model untrained
	actor = (
			keras.models.load_model(args.ckpt_path.parent / "actor.keras")
		if args.load_saved else
			actor_def(state_shape, action_shape, input_setpoint_shape=setpoint_shape)
	)

	lyapunov_model = (
			keras.models.load_model(args.ckpt_path.parent / "lyapunov.keras")
		if args.load_saved else
			V_def(state_shape, input_setpoint_shape=setpoint_shape)
	)
	state_spec = tf.TensorSpec(state_shape)
	setpoint_spec = tf.TensorSpec(setpoint_shape)

	dataset_spec = { "state": state_spec , "setpoint": setpoint_spec}
	dataset = tf.data.Dataset.from_generator(generate_dataset(env), output_signature=dataset_spec)
	batched_dataset = dataset.batch(args.batch_size).take(args.num_batches).cache()
	train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)
