import gymnasium as gym
import numpy as np
import math
import time
import pybullet as p
from pybullet_utils import bullet_client as bc

from simple_carpark.resources.car      import Car
from simple_carpark.resources.plane    import Plane
from simple_carpark.resources.goal     import Goal
from simple_carpark.resources.obstacle import Obstacle

class SimpleCarparkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, isDiscrete=True, renders=False):
        # ─── Action space ──────────────────────────────────────
        if isDiscrete:
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -0.6], dtype=np.float32),
                high=np.array([ 1.0,  0.6], dtype=np.float32),
            )

        # ─── Observation space placeholder (will be updated in reset) ─
        self.observation_space = gym.spaces.Box(
            low = -40.0 * np.ones(12, dtype=np.float32),
            high=  40.0 * np.ones(12, dtype=np.float32),
        )

        # ─── PyBullet client ─────────────────────────────────────
        self._p             = bc.BulletClient(connection_mode=p.GUI if renders else p.DIRECT)
        self._timeStep      = 0.01
        self._actionRepeat  = 50
        self._renders       = renders

        # ─── Episode counters ────────────────────────────────────
        self._step_counter = 0
        self.prev_dist     = None

        # ─── Build the environment ──────────────────────────────
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ─── Reset simulation ────────────────────────────────────
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)

        # ─── Load plane and car ──────────────────────────────────
        Plane(self._p)
        self.car = Car(self._p)

        # ─── Fixed parking slot position ─────────────────────────
        gx, gy = -1.4, -1.1

        # ─── Create obstacles: 2 rows × 3 columns, with center‐front as free slot ─
        slot_z     = 0.1
        slot_width = 0.8
        half_space = slot_width/2 + 0.8
        row_offset = 0.4

        self.barriers = []
        y_positions = [gy + row_offset, gy - row_offset]
        x_offsets  = [-half_space, 0.0, half_space]

        for i, y in enumerate(y_positions):
            for j, x_off in enumerate(x_offsets):
                x = gx + x_off
                # free slot at front center (i=0,j=1)
                if i == 0 and j == 1:
                    self.goal_object = Goal(self._p, (x, y))
                    self.goal_pos    = (x, y, 0)
                else:
                    self.barriers.append(Obstacle(self._p, (x, y, slot_z)))

        # ─── Draw marker for the slot ─────────────────────────────
        self._draw_goal_marker(radius=0.3, segments=32)

        # ─── Initialize metrics ──────────────────────────────────
        car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        gx, gy, _  = self.goal_pos
        self.prev_dist = math.dist(car_pos[:2], (gx, gy))
        self._step_counter = 0

        # ─── Update observation space dimension ─────────────────
        dim = 2 + 2 * len(self.barriers)
        self.observation_space = gym.spaces.Box(
            low  = -40.0 * np.ones(dim, dtype=np.float32),
            high =  40.0 * np.ones(dim, dtype=np.float32),
        )

        return np.array(self.getExtendedObservation(), dtype=np.float32), {}

    def step(self, action):
        # ─── Apply action ────────────────────────────────────────
        if isinstance(self.action_space, gym.spaces.Discrete):
            fwd   = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steer = [-0.6, 0, 0.6] * 3
            action = [fwd[action], steer[action]]
        self.car.apply_action(action)
        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break

        # ─── Get state & compute distance ─────────────────────────
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        gx, gy, _        = self.goal_pos
        dx = car_pos[0] - gx
        dy = car_pos[1] - gy
        dist_to_goal = math.hypot(dx, dy)
        obs = self.getExtendedObservation()

        # ─── Reward: progress, step, goal ─────────────────────────
        # positive progress
        delta   = self.prev_dist - dist_to_goal
        r_prog  = max(-0.1, delta) * 5
        self.prev_dist = dist_to_goal

        # step cost
        r_step  = -0.01

        # orientation check along +X or -X allowed
        _, _, yaw = p.getEulerFromQuaternion(car_orn)
        yaw_norm  = ((yaw + math.pi) % (2*math.pi)) - math.pi
        tol       = 0.25
        orient_ok = (abs(yaw_norm) < tol) or (abs(abs(yaw_norm) - math.pi) < tol)

        # success condition
        pos_ok  = dist_to_goal < 0.3
        success = pos_ok and orient_ok
        r_goal  = 100 if success else 0

        # no early termination on collision; just a small penalty
        collision = any(
            self._p.getClosestPoints(self.car.car, bar.obstacle, distance=0)
            for bar in self.barriers
        )
        r_collision = -1 if collision else 0

        reward = r_prog + r_step + r_collision + r_goal
        reward = max(-50, reward)  # avoid large negative rewards

        # ─── Termination flags ───────────────────────────────────
        terminated = success
        truncated  = self._termination() and not success

        self._step_counter += 1
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, {"success": success}

    def _draw_goal_marker(self, radius=0.4, segments=32):
        z = 0.02
        pts = []
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            pts.append((self.goal_pos[0] + radius * math.cos(theta),
                        self.goal_pos[1] + radius * math.sin(theta), z))
        for i in range(segments):
            self._p.addUserDebugLine(pts[i], pts[i+1], [0,1,0], lineWidth=1)
        # arrow along +X
        head = radius * 1.2
        start = (self.goal_pos[0], self.goal_pos[1], z)
        end   = (self.goal_pos[0] + head, self.goal_pos[1], z)
        self._p.addUserDebugLine(start, end, [1,0,0], lineWidth=2)

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        invPos, invOrn  = self._p.invertTransform(car_pos, car_orn)

        # goal vector
        goal_local, _ = self._p.multiplyTransforms(invPos, invOrn, self.goal_pos, [0,0,0,1])
        obs = [goal_local[0], goal_local[1]]

        # barriers vectors
        for bar in self.barriers:
            pos, _ = self._p.getBasePositionAndOrientation(bar.obstacle)
            loc, _ = self._p.multiplyTransforms(invPos, invOrn, pos, [0,0,0,1])
            obs += [loc[0], loc[1]]

        return obs

    def _termination(self):
        return self._step_counter >= 200

    def close(self):
        self._p.disconnect()