import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import get_linear_fn
from torch.utils.tensorboard import SummaryWriter

from simple_carpark.envs.simple_carpark_env import SimpleCarparkEnv

def make_env():
    env = SimpleCarparkEnv(isDiscrete=True, renders=False)
    return Monitor(env)

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                ep_info = info.get("episode")
                if ep_info is not None:
                    self.writer.add_scalar("episode_reward", ep_info["r"], self.num_timesteps)
                    self.writer.add_scalar("episode_success", int(info.get("success", False)), self.num_timesteps)
        return True

if __name__ == "__main__":
    # ───────────── Create Vectorized Training Environment ─────────────
    num_envs = 4
    env = DummyVecEnv([make_env for _ in range(num_envs)])

    # ───────────── Linear Learning Rate Schedule ─────────────
    lr_schedule = get_linear_fn(1e-3, 1e-5, 100_000)

    # ───────────── Configure DQN Model ─────────────
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        buffer_size=100_000,
        exploration_fraction=0.9,
        exploration_final_eps=0.2,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=5_000,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        tensorboard_log="./carpark_tensorboard/"
    )

    # ───────────── Create Evaluation Environment & Callback ─────────────
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./carpark_tensorboard/best_model",
        log_path="./carpark_tensorboard/eval",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # ───────────── Train the Agent ─────────────
    reward_callback = RewardLoggingCallback(log_dir="./carpark_tensorboard/rewards")
    model.learn(
        total_timesteps=100_000,
        callback=[reward_callback, eval_callback],
        tb_log_name="dqn_minimal"
    )
    model.save("dqn_carpark_minimal")
    print("✅ Model saved to dqn_carpark_minimal.zip")

    # ───────────── Evaluate the Agent Visually ─────────────
    eval_env = make_env()
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        eval_env.render()
    eval_env.close()
