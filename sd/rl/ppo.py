
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim, make
import sd.envs


if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="SetpointedAttitudeEnv-v0")
    parser.set_defaults(test_interval=20480)
    parser.set_defaults(save_summary_interval=20480)
    parser.set_defaults(max_steps=int(2e6))
    parser.set_defaults(horizon=256)
    parser.set_defaults(batch_size=64)
    parser.set_defaults(gpu=-1)
    args = parser.parse_args()

    env = make(args.env_name)
    test_env = make(args.env_name, gui=True, test=True)

    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(32, 32),
        critic_units=(256, 256),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="hard_sigmoid",
        hidden_activation_critic="tanh",
        discount=0.95,
        lam=0.95,
        entropy_coef=0.,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()