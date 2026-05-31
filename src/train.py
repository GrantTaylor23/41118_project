import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Reward function constants for hardcore mode obstacle navigation with jumping
# Forward movement is primary, jumping helps clear obstacles
FORWARD_PROGRESS_REWARD = 1.2  # Strong reward for moving forward
JUMP_POWER_REWARD = 1.1  # Reward for upward velocity (jumping to clear obstacles)
AIRTIME_REWARD = 0.3  # Reward for being airborne
OBSTACLE_CLEARANCE_REWARD = 1.5  # Strongly reward successful obstacle clearing
HEIGHT_GAIN_REWARD = 0.4  # Reward for climbing over obstacles
LANDING_PENALTY = -0.02  # Small penalty for landing
ROTATION_PENALTY = -0.15  # Penalty for excessive rotation
FALL_PENALTY = -0.1  # Penalty for falling (vertical velocity < 0)
CRASH_PENALTY = -50.0  # Penalty for crashing (reduced to allow other signals)
PIT_AVOIDANCE_REWARD = 1.3  # Strong reward for avoiding pits
OBSTACLE_PROXIMITY_PENALTY = -0.15  # Penalty for getting dangerously close to obstacles while grounded
EFFICIENCY_PENALTY = -0.001  # Tiny step cost
OBSTACLE_DETECTION_THRESHOLD = 0.5  # LIDAR distance threshold for obstacle detection
PIT_THRESHOLD = 1.5  # LIDAR distance indicating a pit
DANGER_THRESHOLD = 0.4  # Very close to obstacle - imminent crash zone  

class BipedalRewardWrapper(gym.RewardWrapper):  # ← inherit from gymnasium
    def __init__(self, env):
        super().__init__(env)
        self.prev_height = 0.0
        self.prev_min_lidar = 2.0  # Track previous obstacle distance
        self.prev_leg1_contact = False
        self.prev_leg2_contact = False
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract observations from state vector
        hull_angle = obs[0]           # body angle
        angular_velocity = obs[1]     # rotation speed  
        horizontal_velocity = obs[2]  # forward speed
        vertical_velocity = obs[3]    # jump/fall speed
        hip1_angle = obs[4]
        hip1_speed = obs[5]
        knee1_angle = obs[6]
        knee1_speed = obs[7]
        leg1_ground_contact = obs[8]  # Right leg contact
        hip2_angle = obs[9]
        hip2_speed = obs[10]
        knee2_angle = obs[11]
        knee2_speed = obs[12]
        leg2_ground_contact = obs[13] # Left leg contact
        lidar_readings = obs[14:24]   # 10 lidar sensors looking ahead
        
        reward = 0.0
        
        # ===== OBSTACLE DETECTION =====
        min_lidar = np.min(lidar_readings)
        max_lidar = np.max(lidar_readings)
        
        # Detect if obstacle is close (requires jumping to clear)
        obstacle_close = min_lidar < OBSTACLE_DETECTION_THRESHOLD
        obstacle_moderate = min_lidar < 1.2  # Getting closer to obstacle
        
        # Detect pit (large gap in LIDAR = open space below)
        has_pit = min_lidar > PIT_THRESHOLD
        
        # ===== PRIMARY OBJECTIVE: FORWARD PROGRESS =====
        # Main goal: move forward consistently
        if horizontal_velocity >= 0:
            reward += horizontal_velocity * FORWARD_PROGRESS_REWARD
        else:
            reward += horizontal_velocity * 5.0  # Harsh penalty for moving backward
        
        # ===== JUMPING & OBSTACLE CLEARING =====
        # Reward upward velocity (essential for clearing obstacles)
        if vertical_velocity > 0.1:
            reward += vertical_velocity * JUMP_POWER_REWARD
            
            # Extra bonus for jumping when obstacle is detected
            if obstacle_close:
                reward += vertical_velocity * 0.5  # Additional reward for jumping at right time
        
        # Penalize falling
        if vertical_velocity < -0.1:
            reward += vertical_velocity * FALL_PENALTY
        
        # ===== AIRTIME BONUS =====
        # Reward for being in the air (more control over terrain)
        both_legs_off_ground = (leg1_ground_contact < 0.5) and (leg2_ground_contact < 0.5)
        if both_legs_off_ground:
            reward += AIRTIME_REWARD
        
        # ===== OBSTACLE AVOIDANCE =====
        # Reward for moving away from obstacles (obstacle cleared)
        if min_lidar > self.prev_min_lidar and min_lidar < PIT_THRESHOLD:
            reward += OBSTACLE_CLEARANCE_REWARD
        
        # Pit avoidance: reward for not falling into pits
        # If LIDAR suddenly shows large gap, agent should jump
        if has_pit and both_legs_off_ground and vertical_velocity > 0:
            reward += PIT_AVOIDANCE_REWARD
        
        # Proximity penalty: warn agent it's about to crash into obstacle
        # Penalize being dangerously close to obstacle while on ground (can't jump)
        both_legs_on_ground = (leg1_ground_contact > 0.5) and (leg2_ground_contact > 0.5)
        if min_lidar < DANGER_THRESHOLD and both_legs_on_ground:
            reward += OBSTACLE_PROXIMITY_PENALTY  # Penalize getting stuck near obstacles
        
        # ===== HEIGHT MANAGEMENT =====
        # Reward climbing (going over obstacles)
        if vertical_velocity > 0.05:
            reward += abs(vertical_velocity) * HEIGHT_GAIN_REWARD
        
        # ===== LANDING PENALTY =====
        # Small penalty to encourage continuous jumping pattern
        just_landed = (self.prev_leg1_contact < 0.5 or self.prev_leg2_contact < 0.5) and \
                      (leg1_ground_contact > 0.5 or leg2_ground_contact > 0.5)
        if just_landed:
            reward += LANDING_PENALTY
        
        # ===== ROTATION PENALTY =====
        # Penalize excessive tumbling (keep upright)
        reward -= abs(angular_velocity) * ROTATION_PENALTY
        
        # ===== CRASH PENALTY =====
        # Major penalty if episode terminates (crashed)
        if terminated and not truncated:
            reward += CRASH_PENALTY
        
        # ===== EFFICIENCY PENALTY =====
        reward += EFFICIENCY_PENALTY
        
        # Store current state for next step
        self.prev_min_lidar = min_lidar
        self.prev_leg1_contact = leg1_ground_contact
        self.prev_leg2_contact = leg2_ground_contact
            
        # Reset tracking on episode end
        if terminated or truncated:
            self.prev_height = 0.0
            self.prev_min_lidar = 2.0
            self.prev_leg1_contact = False
            self.prev_leg2_contact = False
            
        return obs, reward, terminated, truncated, info
       

def make_env():
    #return BipedalRewardWrapper(gym.make("BipedalWalker-v3", render_mode="rgb_array"))
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
model = PPO.load("../model/bipedal_ppo_Jumping_V2_hardmode", env=env)
model.learn(total_timesteps=5_000_000)
model.save("../model/bipedal_ppo_Jumping_V2_hardmode_V2")