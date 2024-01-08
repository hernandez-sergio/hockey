
from __future__ import annotations

import glob
import os
import time


from stable_baselines3.ppo import CnnPolicy, MlpPolicy

import pettingzoo
from pettingzoo.utils import BaseWrapper
from pettingzoo.atari import ice_hockey_v2
from pettingzoo.atari import warlords_v3

from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnvStepReturn
import supersuit as ss



class Sb3ShimWrapper(VecEnvWrapper):

    #Necesario para eliminar errores

    metadata = {'render_modes': ['human', 'files', 'none'], "name": "Sb3ShimWrapper-v0"}

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self, seed=None, options=None):
        return self.venv.reset()[0]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
    
    def step(self, actions):
        observation, reward, done, information = super().step(actions)


        for i in range(2):

            if reward[i] == 0.:
                reward[i]=-0.02

            if reward[i] == 1.:
                reward[i] = 10.


        return observation, reward, done, information
    


def train(env_fn, steps: int = 100_000, seed: int | None = 0,**env_kwargs):


    env = env_fn.parallel_env(**env_kwargs)

    print(env.num_agents)


    #Preprocesado. Los wrappers que funcionan con imágenes están comentados a mano ya que se ha comprobado que poner condicionales empeora mucho el rendimiento



    #Se realiza un máximo sobre los últimos 2 frames para evitar la desaparición de objetos
    #env = ss.max_observation_v0(env, 2)

    #Salta frames para una mayor velocidad de procesado
    #env = ss.frame_skip_v0(env, (2,5))

    #Reduce el espacio de observación, ya que no es necesario emplear la totalidad
    #env = ss.resize_v1(env, 84, 84)



    # Para evitar el estricto determinismo, se introduce una probabilidad de evitar la última acción
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)


    # Entrega cuatro frames para poder calcular dirección y sentido de los objetos
    env = ss.frame_stack_v1(env, 4)

    
    
    print(f"Starting training on {str(env.metadata['name'])}.")


    #Para asegurar compatibilidad con SB3
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class='stable_baselines3')

    #Wrapper para solucionar problemas hallados en la ejecución. También se aprovecha para cambiar las recompensas
    env = Sb3ShimWrapper(env)

    env.reset(seed=seed)

    # Use a CNN policy if the observation space is visual
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        batch_size=1024,
    )

    model.learn(total_timesteps=steps)

    model.save(f"DQN_1M_parallel_Rewards")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


if __name__ == "__main__":

    env_fn = ice_hockey_v2

    env_kwargs = dict(obs_type='ram', full_action_space=True, max_cycles=100000, render_mode='human')

    train(env_fn, steps=1_000_000, seed=3, **env_kwargs)