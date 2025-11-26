import os
import hashlib
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", render_mode='human')
        env = Monitor(env) # To give more specific information like avg_rewards, etc...
        return env
    return _init

num_envs = 8
env = SubprocVecEnv([make_env() for _ in range(num_envs)])

log_path = os.path.join("Learnings", "Training", "Logs")
model_save_path = os.path.join("Learnings", "Training", "Saved Models", "PPO_491k_CarRacing_Model")
os.makedirs(log_path, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

ppo_hyperparams = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "tensorboard_log": log_path
}

model = PPO("CnnPolicy", env, **ppo_hyperparams)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=360, verbose=1)

eval_env = gym.make("CarRacing-v3")
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    best_model_save_path=os.path.join("Learnings", "Training", "Best_Models"),
    log_path=os.path.join(log_path, "eval"),
    eval_freq=10000,
    deterministic=True,
    render=False
)

total_timesteps = 491520
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

model.save(model_save_path)

with open(model_save_path + ".zip", "rb") as f:
    file_bytes = f.read()
    hash_value = hashlib.sha256(file_bytes).hexdigest()
    with open('hash_value.txt', 'w') as hash_file:
        hash_file.write(hash_value)

print("Training complete. Model saved at:", model_save_path)
print("SHA-256 hash saved in 'hash_value.txt'.")
