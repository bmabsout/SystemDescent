from gym.envs.registration import register

register(
    id='Pendulum-v1',
    entry_point='sd.envs.Pendulum.Pendulum:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='ModeledPendulum-v1',
    entry_point='sd.envs.Pendulum.ModeledPendulum:ModeledPendulumEnv',
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
    id='CustomBipedalWalker-v0',
    entry_point='sd.envs.bipedal_walker.bipedal_walker:BipedalWalker',
    max_episode_steps=200,
)

register(
    id='ModeledCustomBipedalWalker-v0',
    entry_point='sd.envs.bipedal_walker.modeled_bipedal_walker:ModeledBipedalWalker',
    max_episode_steps=200,
)

register(
    id='AttitudeEnv-v0',
    entry_point='sd.envs.drone.attitude_ctrl_env:AttitudeEnv',
    max_episode_steps=400,
)

register(
    id='SetpointedAttitudeEnv-v0',
    entry_point='sd.envs.drone.attitude_ctrl_env:SetpointedAttitudeEnv',
    max_episode_steps=400,
)

register(
    id='ModeledAttitudeEnv-v0',
    entry_point='sd.envs.drone.modeled_attitude_ctrl_env:ModeledAttitudeEnv',
    max_episode_steps=400,
    kwargs={
        "model_path": "models/AttitudeEnv-v0/5f26ac/checkpoints/checkpoint2"
    }
)

register(
    id='ModeledSetpointedAttitudeEnv-v0',
    entry_point='sd.envs.drone.modeled_attitude_ctrl_env:ModeledSetpointedAttitudeEnv',
    max_episode_steps=50,
    kwargs={
        "model_path": "models/AttitudeEnv-v0/5f26ac/checkpoints/checkpoint2"
    }
)