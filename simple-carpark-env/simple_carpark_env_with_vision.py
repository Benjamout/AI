import gymnasium as gym
import numpy as np
import math
import time
import pybullet as p
from pybullet_utils import bullet_client as bc
import cv2  # <-- for edge detection and image display

from simple_carpark.resources.car      import Car
from simple_carpark.resources.plane    import Plane
from simple_carpark.resources.goal     import Goal
from simple_carpark.resources.obstacle import Obstacle

class SimpleCarparkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, isDiscrete=True, renders=False):
        # ─── Action & placeholder observation space ────────────
        if isDiscrete:
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.Box(
                low = np.array([-1.0, -0.6], dtype=np.float32),
                high= np.array([ 1.0,  0.6], dtype=np.float32),
            )
        self.observation_space = gym.spaces.Box(-40, 40, (12,), dtype=np.float32)

        # ─── PyBullet client setup ─────────────────────────────
        self._p            = bc.BulletClient(connection_mode=p.GUI if renders else p.DIRECT)
        self._timeStep     = 0.01
        self._actionRepeat = 50
        self._renders      = renders

        # ─── Parking counters & dwell logic ────────────────────
        self._step_counter     = 0
        self.prev_dist_to_goal = None
        self._dwell_counter    = 0
        self._dwell_threshold  = 5     # require 5 consecutive low-speed steps
        self._park_lin_tol     = 1   # m/s
        self._park_ang_tol     = 1   # rad/s

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)

        # ─── Build plane, car & randomized bay layout ─────────
        Plane(self._p)
        self.car = Car(self._p)

        gx, gy = -1.4, -1.1
        gx += self.np_random.choice([-0.1, 0, 0.1])
        gy += self.np_random.choice([-0.1, 0, 0.1])

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
                    # self.goal_object = Goal(self._p, (x, y))
                    self.goal_pos    = (x, y, 0)
                else:
                    self.barriers.append(Obstacle(self._p, (x, y, slot_z)))

        self._draw_goal_marker(radius=0.2, segments=32)

        # ─── Init distances & counters ────────────────────────
        car_pos, _             = self._p.getBasePositionAndOrientation(self.car.car)
        self.prev_dist_to_goal = math.dist(car_pos[:2], self.goal_pos[:2])
        self._step_counter     = 0
        self._dwell_counter    = 0

        # ─── Resize observation space to match barriers count ──
        dim = 2 + 2 * len(self.barriers)
        self.observation_space = gym.spaces.Box(
            low  = -40.0 * np.ones(dim, dtype=np.float32),
            high =  40.0 * np.ones(dim, dtype=np.float32),
        )
        return np.array(self.getExtendedObservation(), dtype=np.float32), {}

    def step(self, action):
        # ─── Apply action & step physics ──────────────────────
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

        # ─── Core observations & distance metrics ─────────────
        obs            = self.getExtendedObservation()
        dx_local, dy_local = obs[0], obs[1]
        dist_to_goal      = math.hypot(dx_local, dy_local)

        # ─── Reward components ────────────────────────────────
        # Progress
        if self._step_counter == 0:
            self.prev_dist_to_goal = dist_to_goal
        delta                  = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal
        r_prog                 = max(0, delta) * 10

        # Alignment
        _, car_orn         = self._p.getBasePositionAndOrientation(self.car.car)
        _, _, yaw         = p.getEulerFromQuaternion(car_orn)
        heading_err       = ((yaw + math.pi) % (2*math.pi)) - math.pi
        r_align           = 2.0 * math.cos(heading_err)

        # Collision
        collision = any(
            self._p.getClosestPoints(self.car.car, b.obstacle, distance=0)
            for b in self.barriers
        )
        r_collision = -100 if collision else 0

        # Out-of-bounds
        car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        oob        = not (-3.0 <= car_pos[0] <= 3.0 and -3.0 <= car_pos[1] <= 3.0)
        if oob:
            return np.array(obs, dtype=np.float32), -50, True, False, {"success": False, "oob": True}

        # Parking-dwell & slow-bonus
        lin_vel, ang_vel = self._p.getBaseVelocity(self.car.car)[:2]
        speed_lin        = math.hypot(*lin_vel)
        speed_ang        = abs(ang_vel[2])
        in_zone          = dist_to_goal < 0.2 and abs(heading_err) < 0.26

        if in_zone and speed_lin < self._park_lin_tol and speed_ang < self._park_ang_tol:
            self._dwell_counter += 1
        else:
            self._dwell_counter = 0

        slow_bonus = 0.0
        if in_zone:
            slow_bonus = 5.0 * max(0, (self._park_lin_tol - speed_lin) / self._park_lin_tol)

        parked = (self._dwell_counter >= self._dwell_threshold)
        r_goal = 150 if parked else 0

        # Total reward
        reward            = r_prog + r_align + slow_bonus - 0.5 + r_collision + r_goal
        self._step_counter += 1
        terminated        = parked or collision
        truncated         = self._termination() and not terminated

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, {"success": parked}

    # ─── Existing helper methods (draw, obs-build, termination) ──
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
        head  = radius * 1.2
        start = (self.goal_pos[0], self.goal_pos[1], z)
        end   = (self.goal_pos[0] + head, self.goal_pos[1], z)
        self._p.addUserDebugLine(start, end, [1,0,0], lineWidth=3)

    def getExtendedObservation(self):
        car_pos, car_orn  = self._p.getBasePositionAndOrientation(self.car.car)
        invPos, invOrn    = self._p.invertTransform(car_pos, car_orn)
        goal_local, _     = self._p.multiplyTransforms(invPos, invOrn, self.goal_pos, [0,0,0,1])
        obs = [goal_local[0], goal_local[1]]
        for b in self.barriers:
            pos, _ = self._p.getBasePositionAndOrientation(b.obstacle)
            loc, _ = self._p.multiplyTransforms(invPos, invOrn, pos, [0,0,0,1])
            obs    += [loc[0], loc[1]]
        return obs

    def _termination(self):
        return self._step_counter >= 200

    def close(self):
        self._p.disconnect()

    # ─── New vision methods ─────────────────────────────────
    def get_topdown_image(self, width=256, height=256, camera_height=3.0):
        """
        Grabs a bird’s-eye RGB image centered on the car.
        """
        car_pos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        eye = [car_pos[0], car_pos[1], camera_height]
        tgt = [car_pos[0], car_pos[1], 0]
        up  = [1, 0, 0]
        view  = p.computeViewMatrix(cameraEyePosition=eye,
                                    cameraTargetPosition=tgt,
                                    cameraUpVector=up)
        proj  = p.computeProjectionMatrixFOV(fov=90, aspect=width/height, nearVal=0.1, farVal=10)
        # index 2 is the RGB buffer
        rgb_buf = self._p.getCameraImage(width, height,
                                         viewMatrix=view,
                                         projectionMatrix=proj,
                                         renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        img = np.reshape(rgb_buf, (height, width, 4))[:, :, :3]
        return img

    def simulate_lidar_from_image(self,
                                  image: np.ndarray,
                                  num_beams: int = 360,
                                  camera_height: float = 3.0,
                                  fov_deg: float = 90.0,
                                  max_range: float = 5.0,
                                  ignore_car_radius: float = 0.47):
        """
        Canny-edge + ray-cast in image space → returns `num_beams` ranges in metres.
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        H, W  = edges.shape
        cx, cy = W//2, H//2

        # compute metres-per-pixel
        fov     = math.radians(fov_deg)
        world_h = 2 * camera_height * math.tan(fov/2)
        world_w = world_h * (W/H)
        m_per_px = (world_w/W + world_h/H) * 0.5

        # Mask out car interior
        ignore_px = int(ignore_car_radius / m_per_px)
        mask = np.ones_like(edges, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), ignore_px, 0, -1)
        edges = edges * mask

        angles   = np.linspace(0, 2*math.pi, num_beams, endpoint=False)
        max_r_px = int(max(W, H)/2)

        hits, ranges = [], []
        for θ in angles:
            hit = False
            for r_px in range(max_r_px):
                x = int(cx + r_px * math.cos(θ))
                y = int(cy - r_px * math.sin(θ))
                if x<0 or x>=W or y<0 or y>=H:
                    break
                if edges[y, x]:
                    hits.append((x, y))
                    ranges.append(r_px * m_per_px)
                    hit = True
                    break
            if not hit:
                # no edge → max range
                px = int(cx + max_r_px * math.cos(θ))
                py = int(cy - max_r_px * math.sin(θ))
                hits.append((px, py))
                ranges.append(max_range)
        return np.array(hits), np.array(ranges, dtype=np.float32)


# ─── Demo usage at end of file ──────────────────────────
if __name__ == "__main__":
    env = SimpleCarparkEnv(isDiscrete=True, renders=False)
    obs, _ = env.reset()
    env.step(env.action_space.sample())

    # 1) Capture top-down
    td_img = env.get_topdown_image(width=512, height=512)

    # 2) Edge map
    edges = cv2.Canny(cv2.cvtColor(td_img, cv2.COLOR_RGB2GRAY), 50, 150)

    # 3) Simulate LiDAR
    hits, ranges = env.simulate_lidar_from_image(td_img,
                                                num_beams=360,
                                                max_range=5.0)

    # 5) New—Scatter-only LiDAR overlay, skipping max-range points
    overlay = td_img.copy()
    for (x, y), r in zip(hits, ranges):
        if r < 5.0:  # only draw real hits
            cv2.circle(overlay, (int(x), int(y)), radius=3, color=(0,255,0), thickness=-1)

    # Top-Down
    cv2.namedWindow("Top-Down View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Top-Down View", 700, 700)
    cv2.imshow("Top-Down View", td_img)

    # Edge Map
    cv2.namedWindow("Edge Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edge Map", 700, 700)
    cv2.imshow("Edge Map", edges)

    # LiDAR Overlay
    cv2.namedWindow("LiDAR Hits Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LiDAR Hits Overlay", 700, 700)
    cv2.imshow("LiDAR Hits Overlay", overlay)


    print("Ranges (m):", np.round(ranges, 2))
    print("Press any key in the windows to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

