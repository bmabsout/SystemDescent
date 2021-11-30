import gym
from tf2rl.algos.sac import SAC
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.trainer import Trainer
import sd.envs
import numpy as np

parser = Trainer.get_argument()
parser.set_defaults(logdir="res")
parser = PPO.get_argument(parser)
parser.add_argument("--env", type=str, default="SetpointedAttitudeEnv-v0")

args = parser.parse_args()


env = gym.make(args.env, gui=True)
test_env = gym.make(args.env)
print("obs_space:",env.observation_space)
print("env.action_space.shape[-1]")
# policy = SAC(
#     state_shape=env.observation_space.shape,
#     action_dim=env.action_space.shape[-1],
#     gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
#     memory_capacity=10000,
#     lr=1e-3,
#     # actor_units=[32, 32],
#     # auto_alpha=True,
#     max_action=env.action_space.high,
#     batch_size=64,
#     n_warmup=1000,)

policy = PPO(
    is_discrete=False,
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.shape[-1],
    gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
    # actor_units=[32, 32],
    # auto_alpha=True,
    max_action=env.action_space.high,
    hidden_activation_actor="sigmoid"
)




trainer = Trainer(policy, env, args, test_env=test_env)
# trainer.evaluate_policy()
trainer()
