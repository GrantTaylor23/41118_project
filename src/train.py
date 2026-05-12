import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

FALL_PENALTY = -250 # big penalty for hitting obstacle
GOAL_REWARD = 100    # big reward for reaching goal
STEP_PENALTY = -0.1 # Penalises more steps
PROGRESS_REWARD_SCALE = 5 # multiples negative or positive reward for movement
STILL_ALIVE_REWARD = 0
TOTAL_TIMESTEPS = 500_000
STATIONARY_PENALTY = -5

class BipedalRewardWrapper(gym.RewardWrapper):  # ← inherit from gymnasium

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward += STEP_PENALTY

        forward_velocity = obs[2]   # Hoizontal velocity in observation space
        
        if forward_velocity == 0:
            reward += STATIONARY_PENALTY

        if forward_velocity < 0:
            reward += forward_velocity * PROGRESS_REWARD_SCALE  # Scale punish negative movement
        else:
            reward += forward_velocity * PROGRESS_REWARD_SCALE  # Reward positive horizontal velocity

        if terminated == True:
            reward += FALL_PENALTY # if fall big penalty
        else:
            reward += STILL_ALIVE_REWARD    # currenlty set to 0


        return obs, reward, terminated, truncated, info

def make_env():
    return BipedalRewardWrapper(gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array"))

env = make_vec_env(make_env, n_envs=4)

model = PPO(
    "MlpPolicy",       # MlpPolicy is standard neural network (Multi-Layer Percerption)
    env,
    n_steps=2048,      # steps per env before each update
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.005,    # small entropy bonus encourages exploration
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)

# model = PPO.load("../model/bipedal_ppo", env=env)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("../model/bipedal_ppo")