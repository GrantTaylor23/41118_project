import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

FALL_PENALTY = -250 # big penalty for hitting obstacle
GOAL_REWARD = 100    # big reward for reaching goal
STEP_PENALTY = -0.1 # Penalises more steps
PROGRESS_REWARD_SCALE = 5 # multiples negative or positive reward for movement
STILL_ALIVE_REWARD = 0

class BipedalRewardWrapper(gym.RewardWrapper):  # ← inherit from gymnasium

    def reward(self, reward):
        return reward
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward += STEP_PENALTY

        forward_velocity = obs[2]   # Hoizontal velocity in observation space

        reward += forward_velocity * PROGRESS_REWARD_SCALE  # reward speed
        if forward_velocity < 0:
            reward += forward_velocity * PROGRESS_REWARD_SCALE  # Scale punish negative movement
        else:
            reward += forward_velocity  # Reward positive horizontal velocity

        if terminated == True:
            reward += FALL_PENALTY # if fall big penalty
        else:
            reward += STILL_ALIVE_REWARD    # currenlty set to 0

        if truncated == True:
            reward += GOAL_REWARD

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
    ent_coef=0.05,    # small entropy bonus encourages exploration
    verbose=1
)

model.learn(total_timesteps=100_000)
model.save("../model/bipedal_ppo")