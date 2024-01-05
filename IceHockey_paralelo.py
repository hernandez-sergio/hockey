"""
Uses Stable-Baselines3 to train agents in the Knights-Archers-Zombies environment using SuperSuit vector envs.

This environment requires using SuperSuit's Black Death wrapper, to handle agent death.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time


from stable_baselines3.ppo import CnnPolicy, MlpPolicy

import pettingzoo
from pettingzoo.atari import ice_hockey_v2
from pettingzoo.atari import warlords_v3

from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnvStepReturn
import supersuit as ss

class Sb3ShimWrapper(VecEnvWrapper):
    metadata = {'render_modes': ['human', 'files', 'none'], "name": "Sb3ShimWrapper-v0"}

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self, seed=None, options=None):
        return self.venv.reset()[0]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()


def train(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    print(env.num_agents)
    #Preprocessing
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = ss.max_observation_v0(env, 2)

    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)

    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = ss.frame_skip_v0(env, (2,5))

    # downscale observation for faster processing
    env = ss.resize_v1(env, 84, 84)

    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = ss.frame_stack_v1(env, 4)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    #env = ss.black_death_v3(env)




    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.render_mode=='human'
    # if visual_observation:
    #     # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
    #     #env = ss.color_reduction_v0(env, mode="B")
    #     env = ss.resize_v1(env, x_size=84, y_size=84)
    #     env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)
    
    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = Sb3ShimWrapper(env)

    # Use a CNN policy if the observation space is visual
    model = PPO(
        CnnPolicy if visual_observation else MlpPolicy,
        env,
        verbose=1,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"PPO_250k_parallel")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    #env_fn = knights_archers_zombies_v10
    env_fn = ice_hockey_v2

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(obs_type='grayscale_image', full_action_space=True, max_cycles=100000, render_mode='human')

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, steps=250_000, seed=3, **env_kwargs)

    # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)
