import pybullet as p
import time
import numpy as np

class PD:
    def __init__(self, error=0.0):
        self.prev_error = error
    def pd(self, error):
        error_diff = error - self.prev_error
        self.prev_error = error
        return np.clip(-error*0.4 - error_diff*10, -0.1, 0.1)        


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath("sd/envs/amazingball/assets")
    plate = p.loadURDF("plate.urdf")

    p.setJointMotorControl2(plate, 0, p.POSITION_CONTROL, targetPosition=0, force=5, maxVelocity=2)
    p.setJointMotorControl2(plate, 1, p.POSITION_CONTROL, targetPosition=0, force=5, maxVelocity=2)

    p.setGravity(0, 0, -9.8)
    sphere_body = p.createMultiBody(0.2
        , p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        , basePosition = [0.2,0,0.5]
    )
    p.setRealTimeSimulation(1)
    pd_x = PD()
    pd_y = PD()

    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)
        (x,y,z), orientation = p.getBasePositionAndOrientation(sphere_body)
        
        force_x = pd_x.pd(x)
        force_y = pd_y.pd(y)
        p.setJointMotorControl2(plate, 1, p.POSITION_CONTROL, targetPosition=force_x, force=5, maxVelocity=2)
        p.setJointMotorControl2(plate, 0, p.POSITION_CONTROL, targetPosition=-force_y, force=5, maxVelocity=2)
        # p.setJointMotorControl2(plate, 1, p.POSITION_CONTROL, targetPosition=0, force=5, maxVelocity=2)

    p.disconnect()