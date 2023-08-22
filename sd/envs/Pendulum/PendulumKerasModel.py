import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import dill
from sd import utils

# @tf.function
def pendulum_difference_eq(states, actions):
    g = 10.0
    m = 1.0
    l = 1.0
    max_torque = 2.0
    dt = 0.05
    u = tf.clip_by_value(actions, -max_torque, max_torque)
    max_speed = 8
    cos_th, sin_th, thdot = tf.reshape(states[:, 0], (-1, 1)), tf.reshape(states[:, 1], (-1, 1)), tf.reshape(states[:, 2], (-1, 1))
    
    newthdot = thdot + (3 * g / (2 * l) * sin_th + 3.0 / (m * l**2) * u) * dt
    newthdot = tf.clip_by_value(newthdot, -max_speed, max_speed)    
    th = tf.atan2(sin_th, cos_th)
    newth = newthdot * dt + th
    new_state = tf.stack([tf.cos(newth), tf.sin(newth), newthdot], axis=1)

    return tf.squeeze(new_state, [-1])


# @tf.function
# def keras_lambda(x):
#     return 

def pendulum_diff_Model():
	input_state = keras.Input(shape=(3,))
	input_action = keras.Input(shape=(1,))
	latent_input = keras.Input(shape=(0,))
	inputs = layers.Concatenate()([input_state, input_action, latent_input])
	outputs = layers.Lambda(lambda x: pendulum_difference_eq(x[:, 0:3], x[:, 3:4]))(inputs)
	model = keras.Model(inputs={"state": input_state, "action": input_action, "latent": latent_input}, outputs=outputs)
	model.summary()
	return model



if __name__ == "__main__":
    model = pendulum_diff_Model()
    filepath = utils.random_subdir("models/Pendulum-v2")
    utils.save_checkpoint(model=model, path=filepath, id=0, extra_objs={"pendulum_difference_eq": pendulum_difference_eq})