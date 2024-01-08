from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
import gymnasium as gym

import cv2
import numpy as np

import torch.nn as nn

from gymnasium.envs.registration import EnvSpec, WrapperSpec

class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, frame_stack=4, target_resolution=(72, 48)):
        super(GrayscaleWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(target_resolution[1], target_resolution[0], frame_stack), dtype=np.uint8)
        self.frame_stack = frame_stack
        self.target_resolution = target_resolution
        self.episode_reward = 0

    def observation(self, obs):
        # Warp the frames (crop and resize)
        obs = obs[42:187,32:128]
        obs = cv2.resize(obs, self.target_resolution, interpolation=cv2.INTER_AREA)
        
        # Convert RGB to Grayscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        #cv2.imwrite("test.jpg", gray_obs)

        # Stack frames
        if hasattr(self, 'frames'):
            self.frames = np.roll(self.frames, shift=-1, axis=-1)
            self.frames[:, :, -1] = gray_obs
        else:
            self.frames = np.stack([gray_obs] * self.frame_stack, axis=-1)

        return np.copy(self.frames)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward+=reward
        if terminated:
            if self.episode_reward > 0:
                reward = 10
            self.episode_reward = 0
        return self.observation(observation), reward, terminated, truncated, info


env = GrayscaleWrapper(gym.make("ALE/IceHockey-v5"))
env.reset()
vec_env = DummyVecEnv([lambda: env])
wrapped_env = vec_env.envs[0]
# , , , , , , , user_attrs_gae_lambda_, user_attrs_gamma_, user_attrs_n_steps
# ,,,,,,,,,
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs", 
            gamma= 0.9985168075815646,
            max_grad_norm=4.362513588953114,
            gae_lambda=0.9985736347098447,
            n_steps=1024,
            learning_rate=0.8520071690878388,
            ent_coef=0.04622591539543057,
            policy_kwargs = {
                "net_arch": {"pi": [64], "vf": [64]},
                "activation_fn": nn.ReLU,
                "ortho_init": True,
            }, 
            seed = 5)

model.learn(total_timesteps = 100_000)

# Modify the attribute using the environment's methods or properties
mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=100)
model.save("")
print(mean_reward, std_reward)
env.close()