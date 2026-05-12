from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("../model/bipedal_ppo")
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human") # If hardcore is on here the model will be place in hardcore world

obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()