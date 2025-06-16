from gymnasium.envs.registration import register


register(
    id="DiffRobot-v1",
    entry_point="sd.envs.diff_robot.diff_robot_env:DiffRobotEnv",
    max_episode_steps=500,
    kwargs={"max_vel": 2.0, "dt": 0.05},  # Default parameters
)

register(
    id="ModeledDiffRobot-v1",
    entry_point="sd.envs.diff_robot.Modeled_diff_robot_env:ModeledDiffRobotEnv",
    max_episode_steps=500,
)

register(
    id="Pendulum-v2",
    entry_point="sd.envs.Pendulum.Pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="ModeledPendulum-v2",
    entry_point="sd.envs.Pendulum.ModeledPendulum:ModeledPendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot_continuous-v0",
    entry_point="sd.envs.Acrobot_continuous.Acrobot_continuous:AcrobotEnv",
    max_episode_steps=200,
)

register(
    id="ModeledAcrobot_continuous-v0",
    entry_point="sd.envs.Acrobot_continuous.ModeledAcrobot_continuous:AcrobotEnv",
    max_episode_steps=200,
)

register(
    id="CustomBipedalWalker-v0",
    entry_point="sd.envs.bipedal_walker.bipedal_walker:BipedalWalker",
    max_episode_steps=200,
)

register(
    id="ModeledCustomBipedalWalker-v0",
    entry_point="sd.envs.bipedal_walker.modeled_bipedal_walker:ModeledBipedalWalker",
    max_episode_steps=200,
)

register(
    id="AttitudeEnv-v0",
    entry_point="sd.envs.drone.attitude_ctrl_env:AttitudeEnv",
    max_episode_steps=400,
)

register(
    id="SetpointedAttitudeEnv-v0",
    entry_point="sd.envs.drone.attitude_ctrl_env:SetpointedAttitudeEnv",
    max_episode_steps=400,
)

register(
    id="ModeledAttitudeEnv-v0",
    entry_point="sd.envs.drone.modeled_attitude_ctrl_env:ModeledAttitudeEnv",
    max_episode_steps=400,
    kwargs={"model_path": "models/AttitudeEnv-v0/1ca683/checkpoints/checkpoint9"},
)

register(
    id="ModeledSetpointedAttitudeEnv-v0",
    entry_point="sd.envs.drone.modeled_attitude_ctrl_env:ModeledSetpointedAttitudeEnv",
    max_episode_steps=50,
    kwargs={"model_path": "models/AttitudeEnv-v0/1ca683/checkpoints/checkpoint9"},
)

register(
    id="SetpointedAmazingBallEnv-v0",
    entry_point="sd.envs.amazingball.env:SetpointedAmazingBallEnv",
    max_episode_steps=400,
)
