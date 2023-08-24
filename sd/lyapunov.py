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

def V_def(state_shape: Tuple[int, ...]):
	input_state = keras.Input(shape=state_shape)
	input_setpoint = keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_setpoint])
	dense1 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
	dense2 = layers.Dense(64, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01))(dense1)
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
			obs[2] = obs[2]*7.0

			# the randomization of the setpoint is to force V to be a collection of functions
			# the each parameter (the setpoint) would corresponding to a classic Lyaupnov function
			# in the drone setting, the parameter is the setpoint. 
			# angle = np.where(np.abs(np.cos(init_state)) < 0.7, 0.0, init_state) # randomize setpoints

			angle = np.random.uniform(-np.pi/7.0, np.pi/7.0) + np.random.randint(2)*np.pi
			# chooses only non-sideways angles

			yield {"state":obs, "setpoint": [np.cos(angle), np.sin(angle), 0.0]}
			# yield {"state": obs, "setpoint":[1.0, 0.0, 0.0]} 

			# yield random setpoint. upright or downright
			# yield {"state": obs, "setpoint":[1.0, 0.0, 0.0]} if np.random.randint(2) == 1 else {"state": obs, "setpoint":[-1.0,0.0,0.0]}
	return gen_sample

def save_model(model, name):
	path = Path(args.ckpt_path.parent, name)
	args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
	
	print(str(path))
	model.save(str(path))


pi = tf.constant(math.pi)

@tf.function
def angular_similarity(v1, v2):
    # angular similarity return 1 if v1 == v2
	# only the signal for position, not velocity 
    v1_angle = tf.math.atan2(v1[1], v1[0]) # atan2's range [-pi, pi]
    v2_angle = tf.math.atan2(v2[1], v2[0])
    return tf.abs(tf.abs(v1_angle - v2_angle) - pi) / pi


def train(batches, dynamics_model, actor, V, state_shape, args):
	optimizer=keras.optimizers.Adam(lr=args.lr)
	@tf.function
	def run_full_model(initial_states, set_points, repeat=1):
		'''Runs the dynamics model for repeat steps and returns the final state and the states at each step'''
		# tf.print(initial_states)
		states = tf.TensorArray(tf.float32, size=repeat)
		current_states = initial_states
		for i in range(repeat):
			latent_shape = tuple(current_states.shape[0:1]) + tuple(dynamics_model.input["latent"].shape[1:])
			current_states = dynamics_model(
				{ "state": current_states
				, "action": actor({"state":current_states, "setpoint":set_points})
				, "latent": tf.random.normal(latent_shape)
				}, training=True)  
			states = states.write(i, current_states)
		return current_states, tf.transpose(states.stack(), [1,0,2]) # ok, I think this operation is to put batch back to the first dimension

	@tf.function
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

		# repetitions is a random int, fxu is the final state after that many steps updates.
		# states are a collection of states at each step
		fxu, states = run_full_model(prev_states, set_points,repeat=repetitions)

		# the Lyapunov value at the previous state, i.e. the original state before any update steps as for this batch
		Vx = V({"state": prev_states, "setpoint": set_points}, training=True) # Why V takes setpoint? only reason is to get V(setpoint)==0 ?
		
		# the Lyapunov value at the final state after the update steps
		V_fxu = V({"state": fxu, "setpoint": set_points}, training=True)

		# rational: repetition is necessary for the controller to solve the problem
		# however, the Lyaupnov value should decrease along each every step. The long gap would permit the Lyapunov function 
		# to learn some local minimum
		#V_fxu = V({"state": states[:,0,:], "setpoint": set_points}, training=True)

		# the Lyapunov value at the setpoint(origin) should be zero
		# thus a fully trained V(setpoint) should return zero. Thus zero == 1 when sufficiently trained
		zero = p_mean((1.0-V({"state": set_points, "setpoint": set_points}))**10.0, 0) 
		
		# for condition: V shall decrease along time (i.e. along the update steps)
		# diff = (Vx - V_fxu)

		# another way to define diff: only penalize the negative delta
		diff = (Vx - V_fxu)
		# prev_V = Vx
		# for i in tf.range(repetitions):
		# 	next_V = V({"state": states[:,i,:], "setpoint": set_points}, training=True)
		# 	#diff += tf.nn.leaky_relu(prev_V - next_V, alpha=10**4) / 100
		# 	norm_diff = (prev_V - next_V + 1)/2.0
			
		# 	#diff *= norm_diff**(1.0/repetitions)
			
		# 	# second choice
		# 	diff += (1 - norm_diff)**2 
		# 	prev_V = next_V
		# # second choice 
		# diff = 1 - diff**0.5

		# some reshaping
		transposed_states = tf.transpose(states, [2,0,1])
		transposed_setpoints = tf.broadcast_to(tf.expand_dims(tf.transpose(set_points), axis=-1), tf.shape(transposed_states))
		
		# Q: what is this angular_similarity/close_angle mathematically?
		# A: angles between 
		as_all = angular_similarity(transposed_states, transposed_setpoints)
		angular_similarities = angular_similarity(tf.transpose(fxu)[0:2] ,tf.transpose(set_points)[0:2])
		close_angle = p_mean(angular_similarities, 4.) # 0:2 is for taking position only not the velocity

		# for each sample in the batch, there is a loss for the actor. So the reduce_mean is to get the average loss for the batch?
		# but is actor.loss best to be 0? in this case actor_reg is best to be 1
		# and why is called _reg?
		actor_reg = 1 - tf.tanh(tf.reduce_mean(actor.losses))
		lyapunov_reg = 1 - tf.tanh(tf.reduce_mean(V.losses))

		# what's this? 
		avg_large = tf.minimum(transform(p_mean(Vx,-1.0, slack=1e-15), 0.0, 0.4, 0.0, 1.0), 1.0)

		# if near the setpoint, decrease slower. otherwise decrease faster. This shapes the Lyapunov function. 
		repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
		maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
		line = smooth_constraint(repetitionsf, 1.0, 40, starts_linear=True, to_low = 0.0)
		# under_zero = diff < 0.0
		decreases_everywhere = tf.where(diff < 0.0, 0.5*(1.0+diff), 0.5+0.5*smooth_constraint(diff, 0.0, line, to_low = 0.0, starts_linear=True))
		# non_increasing = p_mean((1.0 + diff[under_zero]), -1.0)**2.0
		# decreasing = p_mean(tf.tanh(3.0*diff[~under_zero]/line), 1.0)
		# tf.minimum(transform(diff, -1.0, line, 0.0, 1.0), 1.0)**2.0
		non_setpoint_Vx = tf.where(angular_similarities > 0.95, 1.0, Vx)
		large_elsewhere = tf.minimum(tf.reduce_min(non_setpoint_Vx)*10 ,1.0)**2
		# gain option I:
		# proof_of_performance = tf.squeeze(tf.reduce_mean(decreases_everywhere))

		# gain option II:
		proof_of_performance = 1 - tf.squeeze(p_mean(1 - decreases_everywhere, 2.0))

		dfl = Constraints(0.0,
			{
			"close_angles": scale_gradient(p_mean(as_all, 2.0), 10.0),
			# "close_angle": smooth_constraint(close_angle,0.65, 0.73),
			"lyapunov": Constraints(0.0, {
				"proof_of_performance": proof_of_performance,
				# "decreasing": decreasing,
				# "non_increasing": non_increasing
				"avg_large": scale_gradient(tf.reduce_mean(tf.minimum(5*non_setpoint_Vx, 1.0))**10, 10.0),
				# "large_elsewhere": large_elsewhere**0.25,
				"zero": scale_gradient(zero, 0.1),
				# "diff": p_mean(diff, -1),
				# "diff": p_mean(diff/2.0 + 0.5, -1.0),
				# "actor_reg": tf.minimum(transform(actor_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
			# 	# "lyapunov_reg": tf.minimum(transform(lyapunov_reg, 0.0, 1.0, 0.0, 1.1), 1.0),
			})
		})

		return dfl

	@tf.function
	def set_gradient_size(gradients, size):
		return size*gradients/tf.norm(gradients)

	@tf.function
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
		
		# tf.print(1-loss_value)
		# tf.print(grads)
		# tf.print(tf.reduce_mean(list(map(lambda x: tf.reduce_mean(tf.abs(x)), grads))))
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

	action_shape = env.action_space.shape
	state_shape = env.observation_space.shape
	dynamics_model = utils.load_checkpoint(args.ckpt_path)
	print()
	print()
	print("dynamics_model:")
	dynamics_model.summary()
	
	# if --load_saved is not set, actor_def and V_def are used to define the actor and lyapunov model untrained
	actor = (
			keras.models.load_model(args.ckpt_path.parent / "actor.keras")
		if args.load_saved else
			actor_def(state_shape, action_shape)
	)

	lyapunov_model = (
			keras.models.load_model(args.ckpt_path.parent / "lyapunov.keras")
		if args.load_saved else
			V_def(state_shape)
	)
	state_spec = tf.TensorSpec(state_shape)
	dataset_spec = { "state": state_spec , "setpoint": state_spec}
	dataset = tf.data.Dataset.from_generator(generate_dataset(env), output_signature=dataset_spec)
	batched_dataset = dataset.batch(args.batch_size).take(args.num_batches).cache()
	train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)
