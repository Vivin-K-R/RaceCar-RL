import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Load model ---
ppo_path = os.path.join('Learnings', 'Training', 'Saved Models', 'PPO_491k_CarRacing_Model')

enviroment_name = 'CarRacing-v3'
env = gym.make(enviroment_name, render_mode='human')
env = DummyVecEnv([lambda: env])  # Wrapper
model = PPO.load(ppo_path, env=env)

# --- Evaluation ---
episodes = 5  # you can change to more
episodic_rewards = []
success_threshold = 500  # define what “success” means for your task

for i in range(1, episodes + 1):
    observation = env.reset()
    done = False
    score = 0.0

    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        score += reward.item()  # convert from numpy array to float

    episodic_rewards.append(score)
    print(f"Episode: {i} | Episodic Reward: {score:.2f}")

# --- Compute Metrics ---
average_episodic_reward = np.mean(episodic_rewards)
success_rate = np.mean(np.array(episodic_rewards) > success_threshold) * 100
training_stability = np.std(episodic_rewards)

# --- Display Summary ---
print("\n===== Evaluation Summary =====")
print(f"Episodes Evaluated: {episodes}")
print(f"Average Episodic Reward: {average_episodic_reward:.2f}")
print(f"Success Rate (> {success_threshold}): {success_rate:.2f}%")
print(f"Training Stability (Std of Rewards): {training_stability:.2f}")
print("================================")

env.close()