import os
import numpy as np
import math
import time
import pybullet as p

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from simple_carpark.envs.simple_carpark_env import SimpleCarparkEnv
from stable_baselines3.common.utils import get_linear_fn

# ───────────── Environment & model ─────────────
env = Monitor(SimpleCarparkEnv(isDiscrete=True, renders=False))

lr_schedule = get_linear_fn(1e-3, 1e-5, 100_000)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=150_000,
    exploration_fraction=0.9,
    exploration_final_eps=0.2,
    train_freq=4,
    target_update_interval=30_000,
    max_grad_norm=10,
    policy_kwargs={"net_arch": [64, 64]},  # just the MLP architecture, no dueling
    verbose=1,
    tensorboard_log="./carpark_tensorboard/"
)

# ───────────── Seed replay buffer with demos ───────
data = np.load("demos.npy", allow_pickle=True)
rb = model.replay_buffer
for obs, action, reward, next_obs, done in data:
    rb.add(
        np.array([obs],      dtype=np.float32),
        np.array([next_obs], dtype=np.float32),
        np.array([action],   dtype=np.int64),
        np.array([reward],   dtype=np.float32),
        np.array([done],     dtype=bool),
        [{}],
    )
print(f"Seeded replay buffer with {len(data)} demo transitions")

# ───────────── Success-rate callback ───────────────
class SuccessLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = 0
        self.episodes  = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("success", False):
                self.successes += 1
            if "episode" in info:
                self.episodes += 1
                if self.episodes % 20 == 0:
                    rate = 100 * self.successes / max(1, self.episodes)
                    print(f"[Episode {self.episodes}] Success Rate: {rate:.1f}%")
                    self.logger.record("rollout/success_rate", rate)
        return True

success_cb = SuccessLoggingCallback()

# ───────────── EvalCallback subclass logging success rate ────────
class EvalWithSuccess(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            rewards, lengths, successes = [], [], []
            for _ in range(self.n_eval_episodes):
                reset_out = self.eval_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

                done = False
                ep_r, ep_len, ep_succ = 0.0, 0, False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_out = self.eval_env.step(action)
                    if len(step_out) == 5:
                        obs, r, terminated, truncated, info = step_out
                    else:
                        obs, r, done_flag, info = step_out
                        terminated, truncated = done_flag, False

                    if isinstance(info, (list, tuple)):
                        info = info[0]

                    ep_r   += r
                    ep_len += 1
                    done    = terminated or truncated
                    ep_succ = ep_succ or info.get("success", False)

                rewards.append(ep_r)
                lengths.append(ep_len)
                successes.append(ep_succ)

            mean_r   = float(np.mean(rewards))
            mean_len = float(np.mean(lengths))
            succ_pct = 100.0 * float(np.mean(successes))

            self.logger.record("eval/mean_reward",    mean_r)
            self.logger.record("eval/mean_ep_length", mean_len)
            self.logger.record("eval/success_rate",   succ_pct)

            if self.verbose > 0:
                print(
                    f"Eval @ {self.num_timesteps}: "
                    f"mean_reward={mean_r:.2f}, "
                    f"mean_length={mean_len:.1f}, "
                    f"success_rate={succ_pct:.1f}%"
                )
        return True

eval_env = Monitor(SimpleCarparkEnv(isDiscrete=True, renders=False))
eval_cb  = EvalWithSuccess(
    eval_env,
    eval_freq=2_000,
    n_eval_episodes=10,
    deterministic=True,
    log_path="./carpark_tensorboard/eval",
    best_model_save_path="./best_model"
)

# ───────────── Train ────────────────
model.learn(
    total_timesteps=200_000,
    callback=[success_cb, eval_cb],
    tb_log_name="dqn_with_demos"
)

model.save("dqn_carpark_with_demos")
print("Model saved to dqn_carpark_with_demos.zip")
