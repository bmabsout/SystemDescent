from gym.envs.registration import register
register(
    id='ModeledPendulum-v0',
    entry_point='sd.envs.ModeledPendulumDir.ModeledPendulum:ModeledPendulumEnv',
    max_episode_steps=200,
)

register(
    id='Acrobot_continuous-v0',
    entry_point='sd.envs.Acrobot_continuous.Acrobot_continuous:AcrobotEnv',
    max_episode_steps=200,
)

register(
    id='ModeledAcrobot_continuous-v0',
    entry_point='sd.envs.Acrobot_continuous.ModeledAcrobot_continuous:AcrobotEnv',
    max_episode_steps=200,
)

register(
    id='AttitudeControlEnv-v0',
    entry_point='sd.envs.drone.attitude_ctrl_env:AttitudeControlEnv',
    max_episode_steps=1000,
)

register(
    id='SqueezedAttitudeControlEnv-v0',
    entry_point='sd.envs.drone.attitude_ctrl_env:SqueezedAttitudeControlEnv',
    max_episode_steps=1000,
)

register(
    id='CustomBipedalWalker-v0',
    entry_point='sd.envs.bipedal_walker.bipedal_walker:BipedalWalker',
    max_episode_steps=200,
)

register(
    id='ModeledCustomBipedalWalker-v0',
    entry_point='sd.envs.bipedal_walker.modeled_bipedal_walker:ModeledBipedalWalker',
    max_episode_steps=200,
)