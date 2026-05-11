import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

OBSTACLE_PENALTY = -25 # big penalty for hitting obstacle
GOAL_REWARD = 100    # big reward for reaching goal
STEP_PENALTY = -0.05 # Penalises more steps
PROGRESS_REWARD_SCALE = 10.1 # multiples negative or positive reward for movement
MINIMUM_SAFE_DISTANCE = 1.0

class BipedalRewardWrapper(gym.RewardWrapper):  # ← inherit from gymnasium
    def reward(self, reward):
        if reward <= -100:
            return -1.0
        return reward
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Now you can use obs, action, AND reward together
        forward_velocity = obs[2]  # BipedalWalker specific
        if reward <= -100:
            reward = -1.0
        else:
            reward += forward_velocity * PROGRESS_REWARD_SCALE  # reward speed

        return obs, reward, terminated, truncated, info

def make_env():
    return BipedalRewardWrapper(gym.make("BipedalWalker-v3", render_mode="rgb_array"))

env = make_vec_env(make_env, n_envs=4)

# # Option 2: pass a factory function (needed when using a wrapper)
# def make_env():
#     return gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")

# env = make_vec_env(make_env, n_envs=4)

# Vectorised envs run multiple copies in parallel — speeds up on-policy training
# env = make_vec_env(env, n_envs=4)

model = PPO(
    "MlpPolicy",       # MlpPolicy is standard neural network (Multi-Layer Percerption)
    env,
    n_steps=2048,      # steps per env before each update
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.005,    # small entropy bonus encourages exploration
    verbose=1
)

model.learn(total_timesteps=100_000)
model.save("../model/bipedal_ppo")