import gymnasium as gym
import gymnasium.spaces
import tensorflow as tf
from sd import dfl


class ModelableEnv(gym.Env):
    '''
    This class is used to learn a model with respect to an environment, requiring a notion of distance between states
    '''
    @staticmethod
    @tf.function
    def closeness_dfl(obs1: tf.Tensor, obs2: tf.Tensor) -> dfl.DFL:
        
        
        abs_diff = tf.abs(obs1 - obs2)
        return 1.0/dfl.p_mean((1.0 + abs_diff), 1.0)

class ModelableWrapper(gym.Wrapper, ModelableEnv):
    pass


def make_modelable(env: gym.Env) -> ModelableEnv:
    '''
    Bubbles up the Modelable Wrapper or adds one if it doesn't exist
    '''
    layer = env
    while env.unwrapped != layer:
        if (isinstance(layer, ModelableEnv)):
            return env
        layer = layer.env
    # no layers are Modelable
    return ModelableWrapper(env)
    