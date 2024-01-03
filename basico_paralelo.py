from pettingzoo.atari import ice_hockey_v2
import supersuit

env = ice_hockey_v2.parallel_env(render_mode="human", obs_type='grayscale_image', full_action_space=True, max_cycles=100000)
observations, infos = env.reset()


#Preprocessing
# as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
# to deal with frame flickering
env = supersuit.max_observation_v0(env, 2)

# repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

# skip frames for faster processing and less control
# to be compatible with gym, use frame_skip(env, (2,5))
env = supersuit.frame_skip_v0(env, 4)

# downscale observation for faster processing
env = supersuit.resize_v1(env, 84, 84)

# allow agent to see everything on the screen despite Atari's flickering screen problem
env = supersuit.frame_stack_v1(env, 4)



while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()