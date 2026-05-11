from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("../model/bipedal_ppo")
env = gym.make("BipedalWalker-v3", render_mode="human")

obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()