import os
import hashlib
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

ppo_path = os.path.join('Learnings', 'Training', 'Saved Models', 'PPO_491k_CarRacing_Model')

with open(ppo_path + ".zip", "rb") as f:
    file_bytes = f.read()
    hash_value_model = hashlib.sha256(file_bytes).hexdigest()
    with open('hash_value.txt', 'r') as hash_file:
        hash_value_saved = hash_file.read()
        if hash_value_model != hash_value_saved:
            raise ValueError("Model integrity check failed! The hash values do not match.")

enviroment_name = 'CarRacing-v3'
env = gym.make(enviroment_name, render_mode='human')

model = PPO.load(ppo_path, env=env)

rew, std = evaluate_policy(model, env, n_eval_episodes=1, render=True)

env.close()

print(rew, std)