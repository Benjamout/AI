# test_render.py
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("r2d2.urdf", [0, 0, 1])

for _ in range(500):
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
