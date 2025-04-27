import numpy as np
import time
import pybullet as p
from simple_carpark.envs.simple_carpark_env import SimpleCarparkEnv  # or your env module

# Discrete‐action key mapping:
# 0=rev-left (Q), 1=rev-straight (S), 2=rev-right (E)
# 3=coast-left (Z), 4=coast-straight (X), 5=coast-right (C)
# 6=fwd-left (A), 7=fwd-straight (W), 8=fwd-right (D)
KEY_ACTION_MAP = {
    ord('c'): 0, ord('x'): 1, ord('z'): 2,
    ord('d'): 3, ord('s'): 4, ord('a'): 5,
    ord('e'): 6, ord('w'): 7, ord('q'): 8,
}

ESCAPE_KEY = 27  # ASCII code for Esc

def teleop_demo(target_successes=20, max_steps_per_episode=200):
    env = SimpleCarparkEnv(isDiscrete=True, renders=True)
    demos = []
    success_count = 0

    obs, _ = env.reset()
    step = 0

    print(f"Tele-op started. Record until {target_successes} successful parks.")
    print("W/A/S/D=drive, Q/E=reverse-turn, Z/X/C=coast. ESC to quit early.")

    while success_count < target_successes:
        keys = env._p.getKeyboardEvents()

        # ESC to quit
        if ESCAPE_KEY in keys and (keys[ESCAPE_KEY] & p.KEY_IS_DOWN):
            print("ESC pressed—exiting tele-op.")
            break

        # choose action
        action = 4  # coast-straight default
        for k, v in keys.items():
            if (v & p.KEY_IS_DOWN) and (k in KEY_ACTION_MAP):
                action = KEY_ACTION_MAP[k]
                break

        # step
        next_obs, reward, terminated, truncated, info = env.step(action)
        demos.append((obs, action, reward, next_obs, terminated or truncated))
        obs = next_obs
        step += 1

        # check for success
        if info.get("success", False):
            success_count += 1
            print(f"Success {success_count}/{target_successes} at step {step}")
            # reset for next demo
            obs, _ = env.reset()
            step = 0
            continue

        # if episode ends otherwise, reset
        if terminated or truncated or step >= max_steps_per_episode:
            print(f"Episode ended (no success). Resetting.")
            obs, _ = env.reset()
            step = 0

    # close and save
    try:
        env.close()
    except Exception:
        pass
    
    demos_array = np.array(demos, dtype=object)
    np.save("demos.npy", demos_array, allow_pickle=True)
    print(f"Saved {len(demos)} transitions to demos.npy")

if __name__ == "__main__":
    teleop_demo()
