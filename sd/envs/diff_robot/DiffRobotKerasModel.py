import tensorflow as tf
from tensorflow import keras
from keras import layers

# import dill
from sd import utils


# @tf.function
def differential_robot_difference_eq(states, actions):
    """
    Differential mobile robot kinematic model
    State: [x, y, theta] - position (x,y) and orientation (theta)
    Actions: [v, w] - linear velocity and angular velocity
    """
    dt = 0.05  # Time step
    max_linear_vel = 2.0  # Maximum linear velocity (m/s)
    max_angular_vel = 2.0  # Maximum angular velocity (rad/s)

    # Extract current state components
    x = tf.reshape(states[:, 0], (-1, 1))  # x position
    y = tf.reshape(states[:, 1], (-1, 1))  # y position
    theta = tf.reshape(states[:, 2], (-1, 1))  # orientation angle

    # Extract and clip control inputs
    v = tf.clip_by_value(
        tf.reshape(actions[:, 0], (-1, 1)), -max_linear_vel, max_linear_vel
    )  # linear velocity
    w = tf.clip_by_value(
        tf.reshape(actions[:, 1], (-1, 1)), -max_angular_vel, max_angular_vel
    )  # angular velocity

    # Differential mobile robot kinematic equations (Euler integration)
    # dx/dt = v * cos(theta)
    # dy/dt = v * sin(theta)
    # dtheta/dt = w

    new_x = x + v * tf.cos(theta) * dt
    new_y = y + v * tf.sin(theta) * dt
    new_theta = theta + w * dt

    # Normalize theta to [-pi, pi] to prevent angle wrap-around issues
    new_theta = tf.atan2(tf.sin(new_theta), tf.cos(new_theta))

    # Stack the new state vector
    new_state = tf.stack([new_x, new_y, new_theta], axis=1)

    return tf.squeeze(new_state, [-1])


def differential_robot_Model():
    """
    Creates a Keras model for differential mobile robot dynamics

    Model Architecture:
    - Input state: [x, y, theta] (3D state vector)
    - Input action: [v, w] (2D control vector)
    - Latent input: placeholder for future extensions
    - Output: next state [x_{k+1}, y_{k+1}, theta_{k+1}]
    """
    input_state = keras.Input(shape=(3,), name="state")  # [x, y, theta]
    input_action = keras.Input(shape=(2,), name="action")  # [v, w]
    latent_input = keras.Input(shape=(0,), name="latent")  # placeholder

    # Concatenate all inputs for the lambda layer
    inputs = layers.Concatenate()([input_state, input_action, latent_input])

    # Apply the difference equation using a Lambda layer
    # The lambda extracts state (first 3 elements) and action (next 2 elements)
    outputs = layers.Lambda(
        lambda x: differential_robot_difference_eq(x[:, 0:3], x[:, 3:5]),
        name="robot_dynamics",
    )(inputs)

    # Create the model with named inputs and outputs
    model = keras.Model(
        inputs={"state": input_state, "action": input_action, "latent": latent_input},
        outputs=outputs,
        name="DifferentialRobotModel",
    )

    model.summary()
    return model


if __name__ == "__main__":
    # Create and save the model
    model = differential_robot_Model()
    filepath = utils.random_subdir("models/DiffRobot-v1")
    utils.save_checkpoint(
        model=model,
        path=filepath,
        id=0,
        extra_objs={
            "differential_robot_difference_eq": differential_robot_difference_eq
        },
    )
