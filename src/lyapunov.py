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
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(inputs)
	dense2 = layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(dense1)
	# dense3 = layers.Dense(128, activation='sigmoid')(dense2)
	outputs = layers.Dense(1, activation='sigmoid')(dense2)
	model = keras.Model(inputs=inputs, outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def friction_actor_def():
	inputs=keras.Input(shape=(3,))
	outputs = layers.Lambda(lambda x: -0.9*x[:,2])(inputs)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model



def pid_actor_def():
	inputs=keras.Input(shape=(3,))
	outputs = layers.Lambda(lambda x: p_norm(x[:,:2]-set_point, 2, axis=0)-0.01*x[:,2])(inputs)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(32, activation='selu', kernel_initializer='lecun_normal')(inputs)
	dense2 = layers.Dense(32, activation='selu', kernel_initializer='lecun_normal')(dense1)
	# dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	prescaled = layers.Dense(np.squeeze(action_shape), activation='tanh')(dense2)
	outputs = prescaled*2.0
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model

# @tf.function
# def geo(l, slack=1e-15,**kwargs):
# 	geo_mean_me = l+slack
# 	problems = tf.reduce_any(geo_mean_me == 0.0, **kwargs)
# 	# Making sure the computation (or gradients) don't NaN when one of the values is 0
# 	v = tf.exp(tf.reduce_mean(tf.math.log(tf.where(geo_mean_me==0.0, 1.0, geo_mean_me)),**kwargs))-slack
# 	return tf.where(problems, 0.0, v)

def generate_dataset(dynamics_model, num_samples):
	# x = np.array([env.observation_space.sample() for _ in range(num_samples)])
	x = np.random.uniform(low=[-np.pi, -7.0], high=[np.pi, 7.0], size=(num_samples, 2))
	res = np.vstack([np.sin(x[:,0]), np.cos(x[:,0]), x[:,1]]).T
	return res.astype(np.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="./saved/823b17/checkpoints/checkpoint20")
parser.add_argument("--num_batches",type=int,default=100)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--batch_size", type=int,default=1000)
parser.add_argument("--epsilon_x", type=float,default=1e-2)
parser.add_argument("--epsilon_diff", type=float,default=1e-2)
parser.add_argument("--lr",type=float, default=1e-4)
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
# actor = friction_actor_def()
lyapunov_model = V_def(state_shape)

def save_model(model, name):
	path = Path(args.ckpt_path, name)
	path.mkdir(parents=True, exist_ok=True)
	print(str(path))
	model.save(str(path))

def train(batches, f, actor, V, state_shape, args):
	optimizer=keras.optimizers.Adam(lr=args.lr)
	@tf.function
	def run_full_model(x, repeat=1):
		states = tf.TensorArray(tf.float32, size=repeat)
		res = x
		for i in range(repeat):
			res = f([res, actor(res, training=True)])
			states = states.write(i, res)
		return res, tf.transpose(states.stack(), [1,0,2])

	@tf.function
	def train_step(x, repetitions, maxRepetitions):
		with tf.GradientTape() as tape:
			fxu, states = run_full_model(x,repeat=repetitions)
			# tf.print(tf.shape(states))
			Vx = V(x, training=True)
			V_fxu = V(fxu, training=True)
			#[-0.866,-0.5,0.0] means pointing 30 degrees to the right
			# (1,0,0) means no movement and pointing upwards
			set_point = tf.constant([[1.0,0.0,0.0]])
			zero = tf.squeeze(1.0-V(set_point))**2.0
			diff = (Vx - V_fxu)
			large = tf.minimum(tf.squeeze(tf.reduce_mean(Vx)),0.1)/0.1
			dist = tf.norm((x - set_point), axis=1)
			# reshaped = tf.reshape(states, tf.concat([[tf.shape(states)[0],  tf.shape(states)[1]*tf.shape(states)[2]]], axis=0))
			ad = angular_similarity(tf.transpose(x), tf.transpose(set_point))
			as_all = angular_similarity(tf.transpose(states, [2,0,1]), tf.transpose(set_point))
			# tf.print(tf.shape(as_all))
			close_angle = p_mean(angular_similarity(tf.transpose(fxu)[0:2] ,tf.transpose(set_point)[0:2]), 3.0)
			be_still = p_mean(1 - tf.abs(tf.transpose(fxu)[2]/7.0) , 1.0)
			normed_dist = tf.math.tanh(dist)
			large2 = tf.math.sigmoid(transform(Vx, 0.0, 0.1, -4.0, 2.0))
			dist_enf_per_elem = tf.where(dist < 0.5, (1.0-Vx), large2)
			small_ones = geo(tf.boolean_mask(1-Vx, dist < 0.5))
			big_ones = p_mean(tf.boolean_mask(tf.minimum(transform(Vx,0.0, 0.3,0.0,1.0),1.0)**2.0, dist >= 0.2),-1.0)
			avg_large = tf.minimum(transform(p_mean(Vx,-1.0, slack=1e-15), 0.0, 0.4, 0.0, 1.0), 1.0)
			# tf.print(Vx)
			# tf.print(tf.reduce_min(dist_enf_per_elem))
			# tf.print(tf.reduce_max(dist_enf_per_elem))
			# tf.print(small_ones)
			# tf.print(big_ones)
			dist_enforcer = geo(dist_enf_per_elem, slack=0.0)
			# tf.print(dist_enforcer)
			# tf.print(tf.reduce_min(- tf.math.softmax(- dist_enf_per_elem)))
			# tf.print(geo(Vx, slack=0.0))
			# tf.print(tf.reduce_sum(tf.where(dist<0.5,Vx,0.0)))
			# tf.print(tf.size(tf.where(dist<0.5)))
			respects_dist = tf.squeeze(tf.reduce_mean(tf.minimum(Vx - normed_dist + 1.0, 1.0)**2.0))
			large_when_far = tf.squeeze(p_mean(tf.where(normed_dist < 0.2, 1.0, tf.math.sigmoid(transform(Vx, 0.0, normed_dist, -8.0, 4.0))), 0, slack=1e-5))
			diff_normed = diff/2. + 0.5
			positive_diff = tf.squeeze(tf.reduce_mean(1.0 + tf.minimum(diff,0.)))
			repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
			maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)
			line = tf.math.tanh(transform(repetitionsf, 0.05, 4.0*maxRepetitionsf*Vx**0.5+0.05, 0.0, 1.0))*Vx**0.5
			# tf.print(repetitionsf)
			# tf.print(Vx)
			# tf.print(line)
			down_everywhere2 = smooth_constraint(diff, 0.0, line)
			# ((diff - 1) *4.0) # 0.98 at the value 1
			# down_everywhere = tf.where(diff < 0.0, 0.0, tf.where(diff >= line, 1.0, tf.math.divide_no_nan(diff, line)))
			# diffg = tf.squeeze(geo(down_everywhere, slack=1e-15))
			diffg_1 = tf.squeeze(p_mean(down_everywhere2, 1.0, slack=1e-5))
			diffg_2 = tf.squeeze(geo(down_everywhere2, slack=1e-3))
			# diffg_1 = tf.squeeze(geo((tf.clip_by_value(diff_normed,0.5, 0.501)-0.5)*1000.0, slack=1e-6))
			# diffb = tf.squeeze(tf.reduce_mean(down_everywhere))
			down = tf.minimum(tf.squeeze(tf.reduce_mean(diff_normed)), 0.55)/0.55
			losses = {
				"zero": zero,
				# "large": args.hyper_psd * tf.reduce_mean(tf.nn.relu(args.epsilon_x - Vx)),
				"large": large,
				# "diff": args.hyper_diff * tf.reduce_mean(tf.nn.relu(args.epsilon_diff*Vx + (V_fxu - Vx) )),
				"positive_diff": positive_diff,
				# "diffg": diffg,
				"diffg_1": transform(diffg_1, 0.0, 1.0, 0.01, 1.0),
				"diffg_2": diffg_2,
				# "diffb": diffb,
				"down": down,
				"respects_dist": respects_dist,
				"large_when_far": large_when_far,
				"dist_enforcer": dist_enforcer,
				"large2": geo(large2),
				"big_ones": big_ones,
				"small_ones": small_ones,
				"avg_large": avg_large,
				"close_angle": smooth_constraint(close_angle, 0.3 + 0.2*repetitionsf/maxRepetitionsf, 0.7 + 0.3*repetitionsf/maxRepetitionsf),
				"close_angle2": smooth_constraint(close_angle, 0.5, 0.7),
				"be_still": transform(be_still, 0.0, 1.0, 0.2, 1.0),
				"close_angles": p_mean(as_all, 3.0),
				"close_angles_smalls": p_mean(as_all, -1.0),
				# "kh": tf.reduce_min(x - set_point, axis=0),
				# "diffg": diffg,
				# "diffg_1": diffg_1,
			}

			# loss_value = losses['diff'] + losses["large"]
			# loss_value = 1-geo(tf.stack([zero, positive_diff, large,diffy]))

			# used_keys = ['close_angles', 'close_angles_smalls', 'be_still']
			used_keys = ['diffg_1', 'zero']
			# loss_value = 1-p_mean(tf.stack([p_mean(tf.stack([close_angle, diffg_1]), 0.5), zero]), 0.1, slack=0)
			# loss_value = 1- p_mean(tf.stack([losses["close_angle2"], losses['diffg_2'], losses['diffg_1'], losses['zero']]), -1.0)#losses['close_angle']
			loss_value = 1- p_mean(tf.stack([losses[u] for u in used_keys]), 1.0)#losses['close_angle']
			# loss_value = 1-geo(tf.stack([diffg_1]), slack=0.0)
			
			# loss_V0 = tf.reduce_mean(V(set_point)**2, keepdims=True)
			# loss_Vx = tf.reduce_mean(tf.nn.relu(args.epsilon_x - Vx), keepdims=True)
			# loss_V_fxu = tf.reduce_mean(tf.nn.relu(args.epsilon_diff + V_fxu - Vx ))
			# loss_value = loss_V0 + args.hyper_psd*loss_Vx + args.hyper_diff*loss_V_fxu
			
		grads = tape.gradient(loss_value, actor.trainable_weights + V.trainable_weights)
		tf.print(1-loss_value)
		# tf.print(tf.reduce_mean(list(map(lambda x: p_mean(x,3.0), filter(lambda x: x != None,grads)))), 1-loss_value)
		optimizer.apply_gradients(zip(grads, actor.trainable_weights + V.trainable_weights))
		# optimizer.minimize(loss_value, actor.trainable_weights + V.trainable_weights).run()
		# train_acc_metric.update_state(y, logits)
		return 1-loss_value, dict(map(lambda k: (k,losses[k]),used_keys))

	@tf.function
	def repeat_train(n, batch):
		maxRepetitions = 40
		repetitions = tf.random.uniform(shape=[], minval=35, maxval=maxRepetitions+1, dtype=tf.dtypes.int32)
		for i in range(n):
			loss_value, losses = train_step(batch, repetitions, maxRepetitions)
		return loss_value, losses

	def train_loop():
		for epoch in range(args.epochs):
			print("\nStart of epoch %d" % (epoch,))
			start_time = time.time()
			for step, batch in enumerate(batches):
				
				loss_value, losses = repeat_train(5, batch)
				# Log every 200 batches.
				if step % 2 == 0:
					print(
						"Training loss (for one batch) at step %d: %.4e, %s"
						% (step, float(loss_value), str(map_dict_elems(lambda v: f"{v:.4e}", losses)))
					)
					print("Seen so far: %d samples" % ((step + 1) * args.batch_size))

			save_model(actor, "actor_tf")
			save_model(lyapunov_model, "lyapunov_tf")
			# Display metrics at the end of each epoch.
			# train_acc = train_acc_metric.result()
			# print("Training acc over epoch: %.4f" % (float(train_acc),))

			# # Reset training metrics at the end of each epoch
			# train_acc_metric.reset_states()

			# # Run a validation loop at the end of each epoch.
			# for x_batch_val, y_batch_val in val_dataset:
			# 	test_step(x_batch_val, y_batch_val)

			# val_acc = val_acc_metric.result()
			# val_acc_metric.reset_states()
			# print("Validation acc: %.4f" % (float(val_acc),))
			print("Time taken: %.2fs" % (time.time() - start_time))
	
	train_loop()

batched_dataset = list(map(lambda _: generate_dataset(dynamics_model, args.batch_size), range(args.num_batches)))
train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)

