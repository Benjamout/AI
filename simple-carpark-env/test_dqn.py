from stable_baselines3 import DQN
from simple_carpark.envs.simple_carpark_env import SimpleCarparkEnv
import time

# ───────────── Load Environment and Model ─────────────
env = SimpleCarparkEnv(isDiscrete=True, renders=True)

# Load trained model
model = DQN.load("dqn_carpark_minimal")

# ───────────── Run One Test Episode ─────────────
obs, _ = env.reset()
done = False
step_count = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step_count += 1
    time.sleep(0.02)  # Optional: slow down for visualisation

env.close()
print(f"Test complete in {step_count} steps. Success: {info.get('success', False)}")
