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
box_half_extents = [1,1,0.1]
corner_coordinates = [-box_half_extents[0], -box_half_extents[1], -box_half_extents[2]]

box = p.createMultiBody(1
    , p.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,0.1])
    , basePosition= [0,0,1]
)


# attach the box to the world in its center
constraint_id = p.createConstraint(box, -1, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [0,0,1])

print(p.getNumJoints(box))

# p.setJointMotorControl2(box, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0, maxVelocity=1, positionGain=0.1, velocityGain=0.1)

# Create a sphere
sphere_body = p.createMultiBody(1
    , p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
    , basePosition = [0,0,2]
)


# Set the initial position and velocity of the sphere
# p.resetBasePositionAndOrientation(sphere_body, , [0,0,0,1])
# p.resetBaseVelocity(sphere_body, [0,0,-1])

# Connect the box to the world with a point-to-point constraint
# p.createConstraint(box_body, -1, -1, p.JOINT_POINT2POINT, np.array([0,0,0], dtype=np.float32), [0,0,0], [0,0,1])


capsule = p.createMultiBody(
    baseMass=1,
    # baseInertialFramePosition=[0,0,0],
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CAPSULE, radius=0.05, height=1),
    basePosition=[1,-1,3]
)

# p.setJointMotorControl2(box, constraint_id, p.VELOCITY_CONTROL, targetVelocity=0.1)
# Set the simulation to run in real time
p.setRealTimeSimulation(1)

# Step the simulation
while True:
    p.stepSimulation()
    time.sleep(0.01)

# Clean up
p.disconnect()
