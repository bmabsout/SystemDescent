from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math
import numpy as np
import os
import uuid
from pathlib import Path
from tqdm import tqdm
import time


def map_dict_elems(fn, d):
    return {k: fn(d[k]) for k in d.keys()}


def to_numpy(tensor):
    return tf.make_ndarray(tf.make_tensor_proto(tensor))

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
    elif p == math.inf:
        return tf.reduce_max(l)
    elif p == -math.inf:
        return tf.reduce_min(l)
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


@tf.function
def andor(l,p):
    return p_mean(tf.stack(l), p)

def latest_subdir(dir="."):
    with_paths = map(lambda subdir: dir + "/" + subdir, os.listdir(dir))
    sub_dirs = filter(os.path.isdir, with_paths)
    return max(sub_dirs, key=os.path.getmtime)

def random_subdir(location):
    uniq_id = uuid.uuid1().__str__()[:6]
    folder_path = Path(location, uniq_id)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path

def save_checkpoint(path, model, id):
    checkpoint_path = Path(path, "checkpoints", f"checkpoint{id}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("saving: ", str(checkpoint_path))
    model.save(str(checkpoint_path))

def latest_model():
    latest_env = latest_subdir("models")
    latest_run = latest_subdir(latest_env)
    latest_checkpoint = latest_subdir(latest_run + "/checkpoints")
    print(f"using model {latest_checkpoint}")
    return latest_checkpoint

def extract_env_name(checkpoint_path):
    return Path(checkpoint_path).parent.parent.parent.name

def inverse_sigmoid(y):
    if y == 0.0:
        return -math.inf
    elif y == 1.0:
        return math.inf
    else:
        return tf.math.log(y/(1-y))


from collections import namedtuple
DFL = namedtuple('DFL', ('operator', 'constraints'))
#DFL stands for Differentiable fuzzy logic
# it is a recursive structure where there are tuples of dictionaries, the first element is the argument to the generalized mean, the second is the definition of the constraints.

def dfl_scalar(dfl):
    return (
            p_mean(tf.stack(list(map(dfl_scalar, dfl.constraints.values()))),dfl.operator)
        if(isinstance(dfl, DFL)) else
            dfl
    )

def format_dfl(dfl):
    if isinstance(dfl, DFL):
        def format_constraint(item):
            name, constraint = item
            return name + f":{format_dfl(constraint)}"
        return f"<{dfl.operator:.2e} {list(map(format_constraint, dfl.constraints.items()))}>"
    elif isinstance(dfl, tf.Tensor):
        return np.array2string(dfl.numpy().squeeze(), formatter={'float_kind':lambda x: f"{x:.2e}"})
    else:
        return str(dfl)

def desc_line():
    desc_line_pb = tqdm(bar_format="[{desc}]")
    def update_description(desc):
        desc_line_pb.update()
        desc_line_pb.set_description(desc)
    return update_description, desc_line_pb

def train_loop(list_of_batches, train_step, end_of_epoch=None):
    for epoch, batches in enumerate(list_of_batches):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()
        with tqdm(batches) as pb:
            update_description, desc_pb = desc_line()
            with desc_pb:
                for batch in pb:
                    update_description(train_step(batch))

        if end_of_epoch:
            end_of_epoch(time.time() - start_time)
        print(f"Time taken: {(time.time() - start_time):.2f}s")