from stable_baselines3 import DQN
from simple_carpark.envs.simple_carpark_env import SimpleCarparkEnv
import time

# ───────────── Load Environment and Model ─────────────
env = SimpleCarparkEnv(isDiscrete=True, renders=True)

# Load trained model
model = DQN.load("./best_model/dqn_carpark_with_demos")
    
# ───────────── Run One Test Episode ─────────────
success_count = 0
for i in range(1000):
    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        # time.sleep(0.02)  # Optional: slow down for visualisation
    
    print(f"Test complete in {step_count} steps. Success: {info.get('success', False)}")
    if info.get("success", True): success_count += 1

env.close()
print(f"Success rate: {success_count / 1000:.2%}")
