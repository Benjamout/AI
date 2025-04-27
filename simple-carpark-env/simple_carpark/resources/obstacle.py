import pybullet as p
import os


class Obstacle:
    def __init__(self, client, base):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simpleobstacle.urdf')
        self.obstacle = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], base[2]])

