from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math
import numpy as np

def map_dict_elems(fn, d):
    return {k: fn(d[k]) for k in d.keys()}


@tf.function
def geo(l, slack=1e-15,**kwargs):
	n = tf.cast(tf.size(l), tf.float32)
	# < 1e-30 because nans start appearing out of nowhere otherwise
	slacked = l + slack
	return tf.reduce_prod(tf.where(slacked < 1e-30, 0., slacked)**(1.0/n), **kwargs) - slack

@tf.function
def p_mean(l, p, slack=0., **kwargs):
	# generalized mean, p = -1 is the harmonic mean, p = 1 is the regular mean, p=inf is the max function ...
	#https://www.wolframcloud.com/obj/26a59837-536e-4e9e-8ed1-b1f7e6b58377
	if p == 0.:
		return geo(tf.abs(l), slack, **kwargs)
	else:
		slacked = tf.abs(l) + slack
		return tf.reduce_mean(tf.where(slacked < 1e-30, 0., slacked)**p, **kwargs)**(1.0/p) - slack

@tf.function
def transform(x, from_low, from_high, to_low, to_high):
	diff_from = tf.maximum(from_high - from_low, 1e-20)
	diff_to = tf.maximum(to_high - to_low, 1e-20)
	return (x - from_low)/diff_from * diff_to + to_low

@tf.function
def inv_sigmoid(x):
	return tf.math.log(x/(1-x))

@tf.function
def smooth_constraint(x, from_low, from_high, to_low=0.03, to_high=0.97):
	return tf.sigmoid(transform(x, from_low, from_high, inv_sigmoid(to_low), inv_sigmoid(to_high)))


pi = tf.constant(math.pi)

@tf.function
def angular_similarity(v1, v2):
	v1_angle = tf.math.atan2(v1[0], v1[1])
	v2_angle = tf.math.atan2(v2[0], v2[1])
	d = tf.abs(v1_angle - v2_angle) % (pi*2.0)
	return 1.0 - transform(pi - tf.abs(tf.abs(v1_angle - v2_angle) - pi), 0.0, pi, 0.0, 1.0)
