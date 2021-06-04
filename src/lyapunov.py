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

def V_def(state_shape: Tuple[int, ...]):
	input_state = keras.Input(shape=state_shape)
	input_setpoint = keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_setpoint])
	dense1 = layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(inputs)
	dense2 = layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(dense1)
	outputs = layers.Dense(1, activation='sigmoid')(dense2)
	model = keras.Model(inputs=[input_state, input_setpoint], outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	input_state = keras.Input(shape=state_shape)
	input_set_point = keras.Input(shape=state_shape)
	inputs = layers.Concatenate()([input_state, input_set_point])
	dense1 = layers.Dense(32, activation='selu', kernel_initializer='lecun_normal')(inputs)
	dense2 = layers.Dense(32, activation='selu', kernel_initializer='lecun_normal')(dense1)
	# dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	prescaled = layers.Dense(np.squeeze(action_shape), activation='tanh')(dense2)
	outputs = prescaled*2.0
	model = keras.Model(inputs=[input_state, input_set_point], outputs=outputs)
	model.summary()
	return model

def generate_dataset(env, num_samples):
	def gen_sample():
		obs = env.reset()
		obs[2] = obs[2]*7.0
		angle = np.random.uniform(-np.pi, np.pi)

		# return [obs, [np.sin(angle), np.cos(angle), 0.0]]
		return [obs, [1.0, 1.0, 0.0]]
	return np.array([gen_sample() for _ in range(num_samples)]).astype(np.float32)

def save_model(model, name):
	path = Path(args.ckpt_path, name)
	path.mkdir(parents=True, exist_ok=True)
	print(str(path))
	model.save(str(path))

def train(batches, f, actor, V, state_shape, args):
	optimizer=keras.optimizers.Adam(lr=args.lr)
	@tf.function
	def run_full_model(initial_states, set_points, repeat=1):
		states = tf.TensorArray(tf.float32, size=repeat)
		current_states = initial_states
		for i in range(repeat):
			current_states = f([current_states, actor([current_states, set_points], training=True)])
			states = states.write(i, current_states)
		return current_states, tf.transpose(states.stack(), [1,0,2])

	@tf.function
	def get_loss(batch, repetitions, maxRepetitions):
		initial_states = batch[:,0,:]
		set_points = batch[:,1,:]
		fxu, states = run_full_model(initial_states, set_points,repeat=repetitions)
		# tf.print(tf.shape(states))
		# set_point = tf.constant([[1.0,0.0,0.0]])
		Vx = V([initial_states, set_points], training=True)
		V_fxu = V([fxu, set_points], training=True)
		#[-0.866,-0.5,0.0] means pointing 30 degrees to the right
		# [1,0,0] means no movement and pointing upwards
		zero = p_mean(1.0-V([set_points, set_points]), 0)**2.0
		diff = (Vx - V_fxu)
		# tf.print(tf.shape(initial_states))
		# tf.print(tf.shape(states))
		# tf.print(tf.shape(set_points))
		# tf.print(tf.shape(states[:,:,:2] - set_points[:,:2]))
		transposed_states = tf.transpose(states, [2,0,1])
		transposed_setpoints = tf.broadcast_to(tf.expand_dims(tf.transpose(set_points), axis=-1), tf.shape(transposed_states))
		as_all = angular_similarity(transposed_states, transposed_setpoints)
		re_all = tf.reshape(as_all, [tf.shape(as_all)[0], tf.shape(as_all)[1],1])
		blurred = tf.nn.conv1d(
			re_all, tf.constant([0.1,0.1,0.2,0.2,0.2,0.1,0.1],shape=[7,1,1]), padding='VALID', data_format="NWC", stride=1
		)[-1]
		close_angle = p_mean(angular_similarity(tf.transpose(fxu)[0:2] ,tf.transpose(set_points)[0:2]), 4.0)


		be_still = p_mean(1 - tf.abs(tf.transpose(fxu)[2]/7.0) , 1.0)
		avg_large = tf.minimum(transform(p_mean(Vx,-1.0, slack=1e-15), 0.0, 0.4, 0.0, 1.0), 1.0)
		repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
		maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
		line = tf.math.tanh(transform(repetitionsf, 0.05, 4.0*maxRepetitionsf*Vx**0.5+0.05, 0.0, 1.0))*Vx**0.5
		down_everywhere2 = smooth_constraint(diff, 0.0, line)
		diffg_1 = tf.squeeze(p_mean(down_everywhere2, -1.0, slack=1e-10))

		losses = {
			"zero": zero,
			"diffg_1": diffg_1**0.2,
			"close_angle": smooth_constraint(close_angle, 0.3 + 0.2*repetitionsf/maxRepetitionsf, 0.7 + 0.3*repetitionsf/maxRepetitionsf),
			"close_angle2": close_angle**2.0,
			"be_still": transform(be_still, 0.0, 1.0, 0.2, 1.0),
			"close_angles": tf.maximum(transform(p_mean(as_all, 3.0), 0.6, 0.7, 0.0, 1.0),0.0),
			"blurred_angles": p_mean(blurred, 3.0),
			# "blurred_angles2": p_mean(blurred, 0.0),
			"avg_large": tf.minimum(p_mean(Vx, -2.0)*2.0, 1.0),
			# "close_angles2": p_mean(as_all, 0.0),
		}

		used_keys = ['close_angles', 'diffg_1', 'zero', 'avg_large']
		# used_keys = ['blurred_angles', 'blurred_angles2']#, 'diffg_1', 'zero']
		loss_value = 1- andor([losses[u] for u in used_keys], -1)
		metrics =  dict(map(lambda k: (k,losses[k]),used_keys))
		return loss_value, metrics

	@tf.function
	def train_step(batch, repetitions, maxRepetitions):
		with tf.GradientTape() as tape:
			loss_value, metrics = get_loss(batch, repetitions, maxRepetitions)
			# loss_value = scale_gradient(loss_value, 1/loss_value**4.0)

		grads = tape.gradient(loss_value, actor.trainable_weights + V.trainable_weights)
		# tf.print(1-loss_value)
		optimizer.apply_gradients(zip(grads, actor.trainable_weights + V.trainable_weights))

		return 1-loss_value, metrics

	@tf.function
	def repeat_train(n, batch):
		maxRepetitions = 15
		repetitions = tf.random.uniform(shape=[], minval=10, maxval=maxRepetitions+1, dtype=tf.dtypes.int32)
		for i in range(n):
			loss_value, metrics = train_step(batch, repetitions, maxRepetitions)
		return loss_value, metrics

	def train_loop():
		for epoch in range(args.epochs):
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
	
	train_loop()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt_path",type=str,default=latest_model())
	parser.add_argument("--num_batches",type=int,default=200)
	parser.add_argument("--epochs",type=int,default=100)
	parser.add_argument("--batch_size", type=int,default=200)
	parser.add_argument("--lr",type=float, default=2e-4)
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

	actor = actor_def(state_shape, action_shape)
	lyapunov_model = V_def(state_shape)

	batched_dataset = [generate_dataset(env, args.batch_size) for _ in range(args.num_batches)]
	train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)

