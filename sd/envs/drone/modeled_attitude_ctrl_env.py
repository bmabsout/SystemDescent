"""
Adapted from 
"""


import time
import numpy as np
from pybullet_utils import bullet_client as bc
import pybullet as p
import sd.envs.drone.signals as signals
from pyquaternion import Quaternion
from sd.envs.drone.assets import load_assets
from sd.envs.drone import attitude_ctrl_env
import keras
from gym import spaces
from sd.rl.utils import FlattenWrapper

def ModeledAttitudeEnv(**kwargs):
    return FlattenWrapper(ModeledMultiAttitudeEnv(**kwargs))

def ModeledSetpointedAttitudeEnv(**kwargs):
    return attitude_ctrl_env.SetpointWrapper(ModeledAttitudeEnv(**kwargs))


class ModeledMultiAttitudeEnv(attitude_ctrl_env.MultiAttitudeEnv):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model = keras.models.load_model(model_path)

    def run_nn(self, obs, actions):
        state = spaces.flatten(self.observation_space, obs)
        latent_shape = (1,) + self.model.input["latent"].shape[1:]
        latent = np.array([[]]) if latent_shape[1] == 0 else np.random.normal(latent_shape)
        new_state = self.model({"state": np.array([state]), "action": actions, "latent": latent}, training=False).numpy()
        return spaces.unflatten(self.observation_space, new_state.squeeze())

    def _physics(self):
        self.obs = self.run_nn(self.obs, self.action)
        # self.ang_v = np.array([[100.0, 0.0, 0.0]])
        # if(self.step_counter % 20 == 0):
        #     print(self.ang_v)
        for drone_id, i in zip(self.DRONE_IDS, range(self.NUM_DRONES)):
            pos, original_quat = self.CLIENT.getBasePositionAndOrientation(drone_id)
            roll = self.obs["ang_v"][i, 0]
            pitch = self.obs["ang_v"][i, 1]
            yaw = self.obs["ang_v"][i, 2]
            previous_quat = Quaternion(np.array(original_quat))
            extrinsic_euler = previous_quat.inverse.rotate(np.array([yaw,roll,pitch]))
            rotated = Quaternion(previous_quat.q)
            rotated.integrate(extrinsic_euler*self.DEG2RAD, self.TIMESTEP)
            self.CLIENT.resetBasePositionAndOrientation(
                drone_id, pos, rotated.q)

    # def _updateAndStoreKinematicInformation(self):
    #     return


if __name__ == "__main__":
    from sd.envs.drone.assets import load_assets

    def sync(i, start_time, timestep):
        """Syncs the stepped simulation with the wall-clock.

        Function `sync` calls time.sleep() to pause a for-loop
        running faster than the expected timestep.

        Parameters
        ----------
        i : int
        Current simulation iteration.
        start_time : timestamp
        Timestamp of the simulation start.
        timestep : float
        Desired, wall-clock step of the simulation's rendering.

        """
        if timestep > .04 or i % (int(1/(24*timestep))) == 0:
            elapsed = time.time() - start_time
            if elapsed < (i*timestep):
                time.sleep(timestep*i - elapsed)

    env = ModeledMultiAttitudeEnv(
        model_path="models/AttitudeEnv-v0/bcab63/checkpoints/checkpoint22",
        gui=True, urdf_path=load_assets.get_urdf_path("cf2x.urdf"), test=True)
    i = 0
    print(env.reset())
    # START = time.time()
    # while(1):
    #     print(env.step(np.array([[0.0, 0.0, 0.0, 0.0]])))
    #     sync(i, START, env.TIMESTEP)
    #     if(i % 100 == 0):
    #         env.render()
    #     i += 1
