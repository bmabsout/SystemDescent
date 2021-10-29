import gym
import tensorflow as tf
import tensorflow.keras.layers as layers
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
from sd.envs.drone.attitude_ctrl_env import AttitudeControlEnv
import numpy as np
from sd.rl import utils

# from sd.envs.drone.assets import load_assets
parser = Trainer.get_argument()
parser.set_defaults(logdir="res")
parser = SAC.get_argument(parser)
args = parser.parse_args()



# env = utils.SqueezeWrapper(AttitudeControlEnv(gui=True))
# test_env = utils.SqueezeWrapper(AttitudeControlEnv())
env = gym.make("SqueezedAttitudeControlEnv-v0", gui=True)
test_env = gym.make("SqueezedAttitudeControlEnv-v0")
policy = SAC(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.shape[-1],
    gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
    memory_capacity=10000,
    lr=3e-4,
    actor_units=[32, 32],
    auto_alpha=True,
    max_action=env.action_space.high,
    batch_size=32,
    n_warmup=500)

trainer = Trainer(policy, env, args, test_env=test_env)
trainer()
