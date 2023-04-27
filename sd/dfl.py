import tensorflow as tf
import numpy as np
from typing import NamedTuple, Union, Dict, Tuple

#handles batches
# def geo(l, slack=0.0):
#     l_arr = np.array(l)
#     if(len(l_arr.shape) == 1): # batch dimension is 0
#         l_arr = np.array([l_arr])
#     slacked = l_arr+slack
#     res = np.zeros(l_arr.shape[0])
#     non_zeros = (slacked > 1e-20).all(axis=1)
#     res[non_zeros] = np.exp(np.mean(np.log(slacked[non_zeros]),axis=1))-slack
#     return res.squeeze()

@tf.function
def geo(l, slack=1e-15,**kwargs): # doesn't handle batches correctly
    # n = tf.cast(tf.size(l), tf.float32)
    # < 1e-30 because nans start appearing out of nowhere otherwise
    slacked = l+slack
    mean_log = tf.reduce_mean(tf.math.log(tf.maximum(slacked, 1e-20)))
    if(tf.reduce_any(slacked < 1e-20)):
        return 0.0
    else:
        return tf.exp(mean_log)-slack
    # return tf.reduce_prod(tf.where(slacked < 1e-30, 0., slacked)**(1.0/n), **kwargs) - slack


@tf.function
def p_mean(l, p, slack=1e-9, **kwargs):
    """
    generalized mean, p = -1 is the harmonic mean, p = 1 is the regular mean, p=inf is the max function ...
    https://www.wolframcloud.com/obj/26a59837-536e-4e9e-8ed1-b1f7e6b58377
    """
    if p == 0.:
        return geo(tf.abs(l), slack, **kwargs) # there's an issue with this
    elif p == np.inf:
        return tf.reduce_max(l)
    elif p == -np.inf:
        return tf.reduce_min(l)
    else:
        slacked = tf.abs(l) + slack
        return tf.reduce_mean(slacked**p, **kwargs)**(1.0/p) - slack

@tf.function
def transform(x, from_low, from_high, to_low, to_high):
    diff_from = tf.maximum(from_high - from_low, 1e-20)
    diff_to = tf.maximum(to_high - to_low, 1e-20)
    return (x - from_low)/diff_from * diff_to + to_low

@tf.function
def inv_sigmoid(x):
    return tf.math.log(x/(1-x))

@tf.function
def smooth_constraint(x, from_low, from_high, to_low=0.03, to_high=0.97, starts_linear=False):
    """like transform but is sigmoidal outside the range instead of linear"""
    scale = lambda x: x*2.0 - 1.0 if starts_linear else x
    unscale = lambda x: (x+1.0)/2.0 if starts_linear else x
    sigmoid_low = inv_sigmoid(unscale(to_low))
    sigmoid_high = inv_sigmoid(unscale(to_high))
    return scale(tf.sigmoid(transform(x, from_low, from_high, sigmoid_low, sigmoid_high)))




class Constraints(NamedTuple):
    '''DFL stands for Differentiable fuzzy logic
        The intention is to build loss functions out of multiple objectives
        it is a recursive structure where there are constraints of constraints, the operator is the argument to the generalized mean, the second is the definition of the constraints.
        '''
    operator: tf.Tensor
    constraints: Dict[str, 'DFL']

DFL = Union[Constraints, tf.Tensor]
# currently specialized to tf tensors, can be made generic if https://bugs.python.org/issue43923 is solved

def dfl_scalar(dfl: DFL):
    return (
            p_mean(tf.stack(list(map(dfl_scalar, dfl.constraints.values()))),dfl.operator)
        if(isinstance(dfl, Constraints)) else
            dfl
    )

def format_dfl(dfl: DFL):
    if isinstance(dfl, Constraints):
        def format_constraint(name_constraint: Tuple[str, DFL]):
            name, constraint = name_constraint
            return f"{name}:{format_dfl(constraint)}"
        return f"<{dfl.operator:.2e} {list(map(format_constraint, dfl.constraints.items()))}>"
    elif isinstance(dfl, tf.Tensor):
        return np.array2string(dfl.numpy().squeeze(), formatter={'float_kind':lambda x: f"{x:.2e}"})
    else:
        return str(dfl)


@tf.function
def implies(normed_pred, normed_implied):
    return normed_pred*normed_implied**(1.0 - normed_pred)

# def and(dfl1: DFL, dfl2: DFL):
#     return Constraints(0, )