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
import time
import argparse

def V_def(state_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(256, activation='sigmoid')(inputs)
	dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	outputs = layers.Dense(1, activation='sigmoid')(dense2)
	model = keras.Model(inputs=inputs, outputs=outputs, name="V")
	print()
	print()
	model.summary()
	return model

def actor_def(state_shape, action_shape):
	inputs = keras.Input(shape=state_shape)
	dense1 = layers.Dense(256, activation='sigmoid')(inputs)
	dense2 = layers.Dense(256, activation='sigmoid')(dense1)
	outputs = layers.Dense(np.squeeze(action_shape), activation='sigmoid')(dense2)
	model = keras.Model(inputs=inputs, outputs=outputs)
	model.summary()
	return model


def tf_geo_mean(l, **kwargs):
	return tf.exp(tf.reduce_mean(tf.math.log(l+1e-10),**kwargs))-1e-10

def geo_mean(l):
	return reduce(lambda a,b: a*b,map(lambda e: (e+1e-10)**(1/len(l)),l)) - 1e-10

def generate_dataset(dynamics_model, num_samples):
	x = np.array([env.observation_space.sample() for _ in range(num_samples)])
	# u = np.array([env.action_space.sample() for _ in range(num_samples)])
	# fxu = dynamics_model.predict([x,u])
	# p = np.repeat([[1,0,0]], num_samples, axis=0) 
	# (1,0,0) means no movement and pointing upwards
	return x

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="./saved/f6bfa5/checkpoints/checkpoint4")
parser.add_argument("--num_batches",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--batch_size", type=int,default=10000)
parser.add_argument("--epsilon_x", type=float,default=.4)
parser.add_argument("--epsilon_diff", type=float,default=1)
parser.add_argument("--lr",type=float, default=1e-2)
parser.add_argument("--hyper_diff", type=float, default=100.0)
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
lyapunov_model = V_def(state_shape)

def map_dict_elems(fn, d):
    return {k: fn(d[k]) for k in d.keys()}

def save_model(model, name):
	path = Path(args.ckpt_path, name)
	path.mkdir(parents=True, exist_ok=True)
	print(str(path))
	model.save(str(path))


def train(train_dataset, f, actor, V, state_shape, args):
	optimizer=keras.optimizers.Adam(lr=args.lr)
	@tf.function
	def train_step(x):
		with tf.GradientTape() as tape:
			action = actor(x, training=True)
			fxu = f([x, action], training=False)
			Vx = V(x, training=True)
			V_fxu = V(fxu, training=True)
			losses = {
				"zero": 1-V(tf.constant([[1.0,0.0,0.0]])),
				# "large": args.hyper_psd * tf.reduce_mean(tf.nn.relu(args.epsilon_x - Vx)),
				"large": tf.reduce_mean(Vx),
				"diff": args.hyper_diff * tf.reduce_mean(tf.nn.relu(args.epsilon_diff*Vx + (V_fxu - Vx) )),
				# "diff": tf.reduce_mean((Vx - V_fxu)/2. + 0.5),
				"diffy": tf.reduce_mean((Vx - V_fxu)**2.0)
			}

			# loss_value = losses['diff'] + losses["large"]
			loss_value = 1-losses["large"]*losses["zero"]
		grads = tape.gradient(loss_value, actor.trainable_weights + V.trainable_weights)
		optimizer.apply_gradients(zip(grads, actor.trainable_weights + V.trainable_weights))
		# optimizer.minimize(loss_value, actor.trainable_weights + V.trainable_weights).run()
		# train_acc_metric.update_state(y, logits)
		return loss_value, losses
	
	# loss_V0 = tf.reduce_mean(V0**2, keepdims=True)
	# loss_Vx = tf.reduce_mean(tf.nn.relu(args.epsilon_x - Vx), keepdims=True)
	# loss_V_fxu = tf.reduce_mean(tf.nn.relu(args.epsilon_diff + Vx - V_fxu ))
	# final_loss = loss_V0 + args.hyper_psd*loss_Vx + args.hyper_diff*loss_V_fxu

	def train_loop():
		for epoch in range(args.epochs):
			print("\nStart of epoch %d" % (epoch,))
			start_time = time.time()
			@tf.function
			def repeat(n):
				for i in range(n):
					loss_value, losses = train_step(x_batch_train)
				return loss_value, losses
			# Iterate over the batches of the dataset.
			for step, x_batch_train in enumerate(train_dataset):
				
				loss_value, losses = repeat(10)
				# Log every 200 batches.
				if step % 20 == 0:
					print(
						"Training loss (for one batch) at step %d: %.8f, %s"
						% (step, float(loss_value), str(map_dict_elems(float, losses)))
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


# @tf.function
# def test_step(x, y):
#     val_logits = model(x, training=False)
#     val_acc_metric.update_state(y, val_logits)

