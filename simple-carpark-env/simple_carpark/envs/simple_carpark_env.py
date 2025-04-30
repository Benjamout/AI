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
        # ─── Action & observation space ─────────────────────────
        if isDiscrete:
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.Box(
                low = np.array([-1.0, -0.6], dtype=np.float32),
                high= np.array([ 1.0,  0.6], dtype=np.float32),
            )
        # placeholder, resized in reset()
        self.observation_space = gym.spaces.Box(-40, 40, (12,), dtype=np.float32)

        # ─── PyBullet client ────────────────────────────────────
        self._p              = bc.BulletClient(connection_mode=p.GUI if renders else p.DIRECT)
        self._timeStep       = 0.01
        self._actionRepeat   = 50
        self._renders        = renders

        # ─── Episode counters & parking‐dwell metrics ───────────
        self._step_counter      = 0
        self.prev_dist_to_goal  = None
        # dwell logic (must sit still for this many steps)
        self._dwell_counter     = 0
        self._dwell_threshold   = 5        # require 10 consecutive low‐speed steps
        self._park_lin_tol      = 0.1      # m/s linear speed threshold
        self._park_ang_tol      = 0.1       # rad/s angular speed threshold

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)

        # ─── Load plane, car, randomized bay centre ─────────────
        Plane(self._p)
        self.car = Car(self._p)

        gx, gy = -1.4, -1.1
        gx += np.random.choice([-0.1, 0, 0.1])
        gy += np.random.choice([-0.1, 0, 0.1])

        slot_z     = 0.1
        slot_width = 0.8
        half_space = slot_width/2 + 0.8
        row_offset = 0.4

        self.barriers = []
        self.goal_pos  = None
        for i, y in enumerate((gy + row_offset, gy - row_offset)):
            for j, x_off in enumerate((-half_space, 0.0, half_space)):
                x = gx + x_off
                if i == 0 and j == 1:
                    self.goal_object = Goal(self._p, (x, y))
                    self.goal_pos    = (x, y, 0)
                else:
                    self.barriers.append(Obstacle(self._p, (x, y, slot_z)))

        self._draw_goal_marker(radius=0.2, segments=32)

        # ─── Initialize distances & counters ────────────────────
        car_pos, _              = self._p.getBasePositionAndOrientation(self.car.car)
        self.prev_dist_to_goal  = math.dist(car_pos[:2], self.goal_pos[:2])
        self._step_counter      = 0
        self._dwell_counter     = 0

        # ─── Resize observation space ───────────────────────────
        dim = 2 + 2 * len(self.barriers)
        self.observation_space = gym.spaces.Box(
            low  = -40.0 * np.ones(dim, dtype=np.float32),
            high =  40.0 * np.ones(dim, dtype=np.float32),
        )

        return np.array(self.getExtendedObservation(), dtype=np.float32), {}

    def step(self, action):
        # ─── Apply action & simulate ────────────────────────────
        if isinstance(self.action_space, gym.spaces.Discrete):
            fwd   = [-1,-1,-1, 0,0,0, 1,1,1]
            steer = [-0.6,0,0.6] * 3
            action = [fwd[action], steer[action]]
        self.car.apply_action(action)
        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                break

        # ─── Observation & local goal vector ───────────────────
        obs = self.getExtendedObservation()
        dx_local, dy_local = obs[0], obs[1]
        dist_to_goal      = math.hypot(dx_local, dy_local)

        # ─── Progress reward ────────────────────────────────────
        if self._step_counter == 0:
            self.prev_dist_to_goal = dist_to_goal
        delta                  = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal
        r_prog                 = max(0, delta) * 10

        # ─── Alignment bonus ────────────────────────────────────
        _, car_orn            = self._p.getBasePositionAndOrientation(self.car.car)
        _, _, car_yaw_world   = p.getEulerFromQuaternion(car_orn)
        heading_err           = ((car_yaw_world + math.pi) % (2*math.pi)) - math.pi
        r_align               = 2.0 * math.cos(heading_err)

        # ─── Collision penalty ──────────────────────────────────
        collision = any(
            self._p.getClosestPoints(self.car.car, b.obstacle, distance=0)
            for b in self.barriers
        )
        r_collision = -100 if collision else 0

        # ─── Out-of-bounds termination ──────────────────────────
        car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        oob        = not (-3.0 <= car_pos[0] <= 3.0 and -3.0 <= car_pos[1] <= 3.0)
        if oob:
            return np.array(obs, dtype=np.float32), -50, True, False, {"success": False, "oob": True}

        # ─── Parking‐dwell logic ─────────────────────────────────
        # check actual velocity in world frame
        lin_vel, ang_vel = self._p.getBaseVelocity(self.car.car)[:2]
        speed_lin        = math.hypot(*lin_vel)
        speed_ang        = abs(ang_vel[2])

        # only count when in goal circle + correctly oriented
        in_zone    = dist_to_goal < 0.2 and abs(heading_err) < 0.26
        low_speed  = (speed_lin < self._park_lin_tol and speed_ang < self._park_ang_tol)
        if in_zone and low_speed:
            self._dwell_counter += 1
        else:
            self._dwell_counter = 0

        # right after you compute speed_lin, speed_ang, and in_zone:
        slow_bonus = 0.0
        if in_zone:
            # linearly scale from 0 → S when speed_lin goes 0 → park_lin_tol
            slow_bonus = 5.0 * max(0, (self._park_lin_tol - speed_lin) / self._park_lin_tol)


        parked      = (self._dwell_counter >= self._dwell_threshold)
        r_goal      = 150 if parked else 0

        # ─── Assemble reward & termination ──────────────────────
        r_step          = -0.2
        reward          = r_prog + r_align + + slow_bonus + r_step + r_collision + r_goal

        self._step_counter += 1
        terminated       = parked or collision
        truncated        = self._termination() and not terminated

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, {"success": parked}

    def _draw_goal_marker(self, radius=0.2, segments=32):
        z   = 0.02
        pts = []
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            pts.append((
                self.goal_pos[0] + radius * math.cos(theta),
                self.goal_pos[1] + radius * math.sin(theta),
                z
            ))
        for i in range(segments):
            self._p.addUserDebugLine(pts[i], pts[i+1], [0,1,0], lineWidth=2)
        head = radius * 1.2
        start = (self.goal_pos[0], self.goal_pos[1], z)
        end   = (self.goal_pos[0] + head, self.goal_pos[1], z)
        self._p.addUserDebugLine(start, end, [1,0,0], lineWidth=3)

    def getExtendedObservation(self):
        car_pos, car_orn  = self._p.getBasePositionAndOrientation(self.car.car)
        invPos, invOrn   = self._p.invertTransform(car_pos, car_orn)
        goal_local, _    = self._p.multiplyTransforms(
            invPos, invOrn, self.goal_pos, [0,0,0,1]
        )
        obs = [goal_local[0], goal_local[1]]
        for b in self.barriers:
            pos, _      = self._p.getBasePositionAndOrientation(b.obstacle)
            loc, _      = self._p.multiplyTransforms(invPos, invOrn, pos, [0,0,0,1])
            obs        += [loc[0], loc[1]]
        return obs

    def _termination(self):
        return self._step_counter >= 200

    def close(self):
        self._p.disconnect()
