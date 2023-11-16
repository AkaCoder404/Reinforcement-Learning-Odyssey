import argparse
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from gymnasium.utils.save_video import save_video
from pettingzoo.mpe import simple_tag_v3


from maddpg import MADDPG

# ENVIRONMENT
MAX_CYCLES              = 100
NUM_GOOD                = 1
NUM_ADVERSARIES         = 3
NUM_OBSTACLES           = 2
CONTINUOUS_ACTIONS      = False
ENV_NAME                = "simple_tag_v3"

EPISODES                = 30

    
print("Creating environment...")
env = simple_tag_v3.parallel_env(
    render_mode="rgb_array",
    max_cycles=MAX_CYCLES,
    num_good=NUM_GOOD,
    num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES,
    continuous_actions=CONTINUOUS_ACTIONS
)
observations, infos = env.reset()

obs_dim_list = [] # observation dimension of each agent
for agent in env.observation_spaces: # continuous observation
    print("Observation space", agent)
    agent_obs_space = env.observation_spaces[agent]
    obs_dim_list.append(agent_obs_space.shape[0]) # Box

act_dim_list = [] # action dimension of each agent
for agent in env.action_spaces: # discrete action
    print("Action space", agent)
    agent_act_space = env.action_spaces[agent]
    act_dim_list.append(agent_act_space.n) # Discrete
    
print("Create MADDPG agent...")
maddpg = MADDPG(obs_dim_list=obs_dim_list, 
            act_dim_list=act_dim_list,
            capacity=0, 
            actor_lr=0,
            critic_lr=0,
            res_dir=None)

model_dir  = os.path.join('results', ENV_NAME, "run_4")
assert os.path.exists(model_dir)
data = torch.load(os.path.join(model_dir, 'model.pt'))
for agent, actor_parameter in zip(maddpg.agents, data):
    agent.actor.load_state_dict(actor_parameter)
print(f'MADDPG load model.pt from {model_dir} successful...')

print("Start evaluating")
total_reward = np.zeros((EPISODES, env.num_agents))            # reward of each episode
for episode in range(EPISODES):
    obs, infos = env.reset()
    obs = [obs[agent] for agent in env.agents]                 # get observation for each agent
    episode_reward = np.zeros((MAX_CYCLES, env.num_agents))    # record reward of each agent in this episode

    frames = []
    total_steps_taken = 0
    for step in range(MAX_CYCLES):  # interact with the env for an episode
        frames.append(env.render())
        
        actions_list = maddpg.select_action(obs)
        actions_dict = {agent: actions_list[n] for n, agent in enumerate(env.agents)}
        actions_desc = {agent: np.argmax(actions_dict[agent]) for agent in env.agents}
        
        next_observations, rewards, terminations, truncations, infos = env.step(actions_desc)
        next_observations = [next_observations[agent] for agent in next_observations.keys()]
        rewards_list = [rewards[agent] for agent in rewards.keys()]
        episode_reward[step] = rewards_list
        env.render()
        # time.sleep(0.1)
        obs = next_observations
        total_steps_taken += 1
        

    # episode finishes, calculate cumulative reward of each agent in this episode
    cumulative_reward = episode_reward.sum(axis=0)
    total_reward[episode] = cumulative_reward
    print(f'episode {episode + 1}: cumulative reward: {cumulative_reward}, total steps: {total_steps_taken}')
    
save_video(
    frames, 
    os.path.join(model_dir), 
    fps=env.metadata['render_fps'],
    name_prefix=f'mpe_good_{NUM_GOOD}_adv_{NUM_ADVERSARIES}_obs_{NUM_OBSTACLES}',
)

# all episodes performed, evaluate finishes
print("Plot total rewards...")
fig, ax = plt.subplots()
x = range(1, EPISODES + 1)
for agent in range(env.num_agents):
    ax.plot(x, total_reward[:, agent], label=agent)
    # ax.plot(x, get_running_reward(total_reward[:, agent]))
ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('reward')
title = f'evaluating result of maddpg solve {ENV_NAME}'
ax.set_title(title)
plt.savefig(os.path.join(model_dir, title))
