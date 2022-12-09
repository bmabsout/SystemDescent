import pybullet as p
import time
import numpy as np

# Create a simulation
physicsClient = p.connect(p.GUI)

# Set the gravity
p.setGravity(0,0,-10)

# Add a plane to serve as the ground
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)

# Create a thin box
box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,0.1])
box_body = p.createMultiBody(1, box_id)

# Set the initial position of the box
p.resetBasePositionAndOrientation(box_body, [0,0,0], [0,0,0,1])

# Create a sphere
sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
sphere_body = p.createMultiBody(1, sphere_id)

# Set the initial position and velocity of the sphere
p.resetBasePositionAndOrientation(sphere_body, [0,0,2], [0,0,0,1])
p.resetBaseVelocity(sphere_body, [0,0,-1])

# Connect the box to the world with a point-to-point constraint
# p.createConstraint(box_body, -1, -1, p.JOINT_POINT2POINT, np.array([0,0,0], dtype=np.float32), [0,0,0], [0,0,1])

constraint_id = p.createConstraint(box_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0,0,1])

# p.setJointMotorControl2(box_body, constraint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0, maxVelocity=1, positionGain=0.1, velocityGain=0.1, lowerLimit=0, upperLimit=0, axisMask=[1,1,0])

# p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, targetVelocity=(deltav + self.state[3]))
# Set the simulation to run in real time
p.setRealTimeSimulation(1)

# Step the simulation
while True:
    p.stepSimulation()
    time.sleep(0.01)

# Clean up
p.disconnect()
