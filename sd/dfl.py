import tensorflow as tf
import numpy as np
from typing import NamedTuple, Union, Dict, Tuple


@tf.function
def tf_pop(tensor, axis):
    return tf.concat([tf.slice(tensor, [0], [axis]), tf.slice(tensor, [axis+1], [-1])], 0)

@tf.custom_gradient
def adaptive_grads(x, y):
    def grad(dy):
        return dy, tf.zeros_like(y)
    return x, grad


@tf.function
def p_mean(l, p: float, slack=1e-15, default_val=tf.constant(0.0), axis=None):
    """
    The Generalized mean
    l: a tensor of elements we would like to compute the p_mean with respect to, elements must be > 0.0
    p: the value of the generalized mean, p = -1 is the harmonic mean, p = 1 is the regular mean, p=inf is the max function ...
    slack: allows elements to be at 0.0 with p < 0.0 without collapsing the pmean to 0.0 fully allowing useful gradient information to leak
    axis: axis or axese to collapse the pmean with respect to, None would collapse all
    https://www.wolframcloud.com/obj/26a59837-536e-4e9e-8ed1-b1f7e6b58377
    """
    p = tf.cast(p, tf.float32)
    l = tf.cast(l, tf.float32)
    p = tf.where(tf.abs(tf.cast(p, tf.float32)) < 1e-5, -1e-5 if p < 0.0 else 1e-5, p)

    return tf.cond(tf.reduce_prod(tf.shape(l)) == 0 # condition if an empty array is fed in
        , lambda: tf.broadcast_to(default_val, tf_pop(tf.shape(l), axis)) if axis else default_val
        , lambda: tf.reduce_mean((l + slack)**p, axis=axis))**(1.0/p) - slack
        # tf.debugging.assert_greater_equal(l, 0.0)


@tf.function
def i_mean(l, p, **kwargs):
    """inverse of p_mean, changes focus point from 0.0 to 1.0 (how far you are from 1 rather than 0)"""
    return 1.0 - p_mean(1.0 - l, p, **kwargs)

# build_piecewise([(-1.0, 0.0), (-0.1, 0.1), (0.0, 0.2), (line, 0.9), (1.0, 1.0)])
# tf.where(val < -0.1, transform(val, -1.0, -0.1, 0.0, 0.1), tf.where(val < 0.0, transform(val, -0.1, 0.0, 0.1, 0.2), tf.where(val < line, transform(val, 0.0, line, 0.2, 0.9), transform(val, line, 1.0, 0.9, 1.0))))
@tf.function
def transform(x, from_low, from_high, to_low, to_high, clipped=False):
    diff_from = tf.maximum(from_high - from_low, 1e-20)
    diff_to = tf.maximum(to_high - to_low, 1e-20)
    mapped = (x - from_low)/diff_from * diff_to + to_low
    return tf.clip_by_value(mapped, to_low, to_high) if clipped else mapped

@tf.function
def build_piecewise(xy, val, clipped=False):
        items = reversed(list(zip(xy, xy[1:])))
        (prev_x, prev_y), (x, y) = next(items)
        final = transform(val, prev_x, x, prev_y, y, clipped=clipped)
        # str = f"transform(val, {prev_x}, {x}, {prev_y}, {y})"
        for (prev_x, prev_y), (x, y) in items:
            final = tf.where(val < x, transform(val, prev_x, x, prev_y, y, clipped=clipped), final)
            # str = f"tf.where(val < {x}, transform(val, {prev_x}, {x}, {prev_y}, {y}), {str})"
        # print(str)
        return final


@tf.function
def inv_sigmoid(x):
    return tf.math.log(x/(1-x))

@tf.function
def smooth_constraint(x, from_low, from_high, to_low=0.03, to_high=0.97, starts_linear=False, clipped=False):
    """like transform but is sigmoidal outside the range instead of linear"""
    scale = lambda x: x*2.0 - 1.0 if starts_linear else x
    unscale = lambda x: (x+1.0)/2.0 if starts_linear else x
    sigmoid_low = inv_sigmoid(unscale(to_low))
    sigmoid_high = inv_sigmoid(unscale(to_high))
    return scale(tf.sigmoid(transform(x, from_low, from_high, sigmoid_low, sigmoid_high, clipped=clipped)))


def tensor_to_str(tensor: tf.Tensor):
    return np.array2string(tensor.numpy().squeeze(), formatter={'float_kind':lambda x: f"{x:.2e}"})

DFL = Union["Constraints", tf.Tensor]

def format_constraint(name_constraint: Tuple[str, DFL]):
    name, constraint = name_constraint
    constraint_str = (tensor_to_str if isinstance(constraint, tf.Tensor) else str)(constraint)
    return f"{name}:{constraint_str}"

class Constraints(NamedTuple):
    '''DFL stands for Differentiable fuzzy logic
        The intention is to build loss functions out of multiple objectives
        it is a recursive structure where there are constraints of constraints, the operator is the argument to the generalized mean, the second is the definition of the constraints.
        '''
    operator: tf.Tensor
    constraints: Dict[str, 'DFL']
    def scalarize(self):
        return p_mean(tf.stack(list(map(dfl_scalar, self.constraints.values()))), self.operator, default_val=1.0)
    def __str__(self):
        return f"{self.operator:.2g}<{' '.join(map(format_constraint, self.constraints.items()))}>"

class InvConstraints(Constraints):
    def scalarize(self):
        return 1.0 - p_mean(tf.stack(list(map(lambda x: 1.0 - dfl_scalar(x), self.constraints.values()))),self.operator, default_val=1.0)
    def __str__(self):
        return f"{self.operator:.2g}<!{' '.join(map(format_constraint, self.constraints.items()))}!>"

# currently specialized to tf tensors, can be made generic if https://bugs.python.org/issue43923 is solved

def dfl_scalar(dfl: DFL):
    return (
            dfl.scalarize()
        if(isinstance(dfl, Constraints)) else
            dfl
    )


@tf.function
def implies(normed_pred, normed_implied):
    return normed_pred*normed_implied**(1.0 - normed_pred)

# def and(dfl1: DFL, dfl2: DFL):
#     return Constraints(0, )

@tf.function
def laplace_smoothing(weaken_me, weaken_by):
    return (weaken_me + weaken_by)/(1.0 + weaken_by)


@tf.custom_gradient
def scale_gradient(x, scale):
  def grad(dy): return (dy * scale, None)
  return x, grad


@tf.keras.utils.register_keras_serializable(package='Custom', name='move_toward_zero')
@tf.custom_gradient
def move_toward_zero(x):
    #tweaked to be a good activity regularizer for tanh within the dfl framework
    def grad(dy):
        return -dy*x*x*x*5.0
    return tf.sigmoid(-tf.abs(x)+5), grad