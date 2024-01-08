import gymnasium

from stable_baselines3 import PPO
import os

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from typing import Any
from typing import Dict

import torch
import torch.nn as nn
import cv2
import numpy as np

class GrayscaleWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, frame_stack=4, target_resolution=(72, 48)):
        super(GrayscaleWrapper, self).__init__(env)
        #self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(target_resolution[1], target_resolution[0], frame_stack), dtype=np.uint8)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(128,), dtype=np.uint8)
        self.frame_stack = frame_stack
        self.target_resolution = target_resolution
        self.episode_reward = 0

    def observation(self, obs):
        return obs

    '''def observation(self, obs):
        # Warp the frames (crop and resize)
        obs = obs[42:187,32:128]
        obs = cv2.resize(obs, self.target_resolution, interpolation=cv2.INTER_AREA)
        
        # Convert RGB to Grayscale
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # Stack frames
        if hasattr(self, 'frames'):
            self.frames = np.roll(self.frames, shift=-1, axis=-1)
            self.frames[:, :, -1] = gray_obs
        else:
            self.frames = np.stack([gray_obs] * self.frame_stack, axis=-1)

        return np.copy(self.frames)'''
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward+=reward
        if terminated:
            if self.episode_reward > 0:
                reward = 10
            self.episode_reward = 0
        return self.observation(observation), reward, terminated, truncated, info

ALGORTIHM="PPO"
models_dir = "hockey/models/" + ALGORTIHM
log_dir = "hockey/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

N_STEPS = 100
N_TRIALS = 25
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 10
N_TIMESTEPS = 100_000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 100

ENV_ID = "ALE/IceHockey-v5"

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": GrayscaleWrapper(gymnasium.make(ENV_ID, obs_type='ram')),
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            print("Test")
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(**kwargs)
    # Create env used for evaluation.
    eval_env = Monitor(GrayscaleWrapper(gymnasium.make(ENV_ID, obs_type='ram')))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    study.trials_dataframe().to_csv('study.csv',index=False)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))