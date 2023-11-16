"""
Implementing MADDPG with PettingZoo MPE and PyTorch

# print(f"Agent {agent} taking action {action}")        
# action space: [no_action, move_left, move_right, move_down, move_up], (5, )
# action 0 is no action
# action 1 is move left
# action 2 is move right
# action 3 is move down
# action 4 is move up
# observ space: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities], (16, )
# (1, 2, n * 2, n * 1), n is number of other agents
"""

from pettingzoo.mpe import simple_adversary_v3, simple_v3, simple_tag_v3
import time
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from maddpg import MADDPG

# env = simple_v3.env(max_cycles=25, continuous_actions=False)
# env = simple_adversary_v3.env(render_mode="human")
# env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)


# ENVIRONMENT SETTINGS
MAX_CYCLES              = 25
NUM_GOOD                = 1
NUM_ADVERSARIES         = 3
NUM_OBSTACLES           = 2
CONTINUOUS_ACTIONS      = False
ENV_NAME                = "simple_tag_v3"

# TRAINING SETTINGS
BUFFER_CAPACITY         = int(1e6)
ACTOR_LR                = 1e-2
CRITIC_LR               = 1e-2
BATCH_SIZE              = 1024
TAU                     = 0.02
GAMMA                   = 0.95
LEARN_INTERVAL          = 100
SAVE_INTERVAL           = 100
STEPS_BEFORE_LEARN      = 5e4
EPISODES                = 30000


# create folder to save result
print("Creating folder to save results...")
env_dir = os.path.join('results', ENV_NAME)
if not os.path.exists(env_dir):
    os.makedirs(env_dir)
total_files = len([file for file in os.listdir(env_dir)])
res_dir = os.path.join(env_dir, f'run_{total_files + 1}')
os.makedirs(res_dir)
model_dir = os.path.join(res_dir, 'model')
os.makedirs(model_dir)

print("Creating environment...")
count = 0
env = simple_tag_v3.parallel_env(
    max_cycles=MAX_CYCLES,
    num_good=NUM_GOOD,
    num_adversaries=NUM_ADVERSARIES,
    num_obstacles=NUM_OBSTACLES,
    continuous_actions=CONTINUOUS_ACTIONS
)
observations, infos = env.reset()

# Environment detailss
print("-" * 50)
print("Environment Details")
print("Observations shape", observations)
print("Infos", infos)
print("Action space", env.action_space)
print("Agents", env.agents)
print(env.metadata)

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
    
print("Observation dimensions", obs_dim_list)
print("Action dimensions", act_dim_list)
print("-" * 50)

def print_step_info(observations, rewards, terminations, truncations, infos):
    print("*" * 100)
    print("Step information")
    print("Observations: ", observations)
    print("Rewards: ", rewards)
    print("Terminations: ", terminations)
    print("Truncations: ", truncations)
    print("Infos: ", infos)
    print("*" * 100)
    
# Create MADDPG agent
maddpg = MADDPG(obs_dim_list=obs_dim_list, 
                act_dim_list=act_dim_list,
                capacity=BUFFER_CAPACITY, 
                actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR,
                res_dir=None)

# Episodes
# start = time()
total_reward = np.zeros((EPISODES, env.num_agents)) # reward of each agent in each episode
total_steps = 0
for episode in range(EPISODES):
    obs, infos = env.reset()
    obs = [obs[agent] for agent in env.agents] # make obs into a list

    # Cycles
    episode_reward = np.zeros((MAX_CYCLES, env.num_agents)) # reward of each agent in each episode
    # print("Episode_reward", episode_reward.shape)
    step = 0
    while env.agents: # 
        # this is where you would insert your policy
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents} # Random action
        # print(f"{count}: Actions", actions)
    
        actions_list = maddpg.select_action(obs)
        # Turn actions into a dictionary
        actions_dict = {agent: actions_list[n] for n, agent in enumerate(env.agents)}
        # Make actions discrete
        actions_desc = {agent: np.argmax(actions_dict[agent]) for agent in env.agents}
        

        next_observations, rewards, terminations, truncations, infos = env.step(actions_desc)
        # next_observations = [next_observations[agent] for agent in env.agents]
        next_observations = [next_observations[agent] for agent in next_observations.keys()]
        

        # Turn rewards into a list from a dictionary
        # assert [rewards[agent] for agent in env.agents] == [rewards[agent] for agent in rewards.keys()]
        # rewards_list = [rewards[agent] for agent in env.agents]
        rewards_list = [rewards[agent] for agent in rewards.keys()]
        # print(f"{step}: Rewards list", rewards_list)
        episode_reward[step] = rewards_list
        
        # if total_steps > EPISODES / 10 * 25:
        #     env.render()

        # Turn terminations into a list from a dictionary
        # terminations = [terminations[agent] for agent in env.agents]
        terminations = [terminations[agent] for agent in terminations.keys()]
    
        # Add experience to buffer
        maddpg.add_experience(obs, actions_list, rewards_list, next_observations, terminations)
        # only start to learn when there are enough experiences to sample
        if total_steps > STEPS_BEFORE_LEARN:
            if total_steps % LEARN_INTERVAL == 0:
                maddpg.learn(BATCH_SIZE, GAMMA)
                maddpg.update_target(TAU)
            if episode % SAVE_INTERVAL == 0:
                torch.save([agent.actor.state_dict() for agent in maddpg.agents],
                            os.path.join(model_dir, f'model_{episode}.pt'))

        obs = next_observations
        
        # if total_steps > (EPISODES - EPISODES / 10) * MAX_CYCLES:
        #     print("Total steps", total_steps)
        #     # sleep for 0.01
        #     time.sleep(0.1)
            
        total_steps += 1
        step += 1
        
    # Episode finishes
    # Calculate cumulative reward of each agent in this episode
    cumulative_reward = episode_reward.sum(axis=0)
    total_reward[episode] = cumulative_reward
    
    if episode == EPISODES / 10:
        print(f'Episode {episode + 1}: cumulative reward: {cumulative_reward}, 'f'sum reward: {sum(cumulative_reward)}')
    

    
# Plot rewards
print("Plotting rewards...")
print("Total reward", len(total_reward))
print("Total reward shape", total_reward.shape)

# Plot rolling average
window_size = 100
plt.plot(total_reward)
for i in range(NUM_GOOD + NUM_ADVERSARIES):
    # Get (1000, 0)
    agent_total_rewards = total_reward[:, i]
    rolling_average = np.convolve(agent_total_rewards, np.ones(window_size), "valid") / window_size
    plt.plot(rolling_average)
    
plt.title("Rewards")
plt.savefig("rewards.png")
plt.show()
plt.clf()
    
print("Saving results...")
# Save agent parameters and training reward
torch.save([agent.actor.state_dict() for agent in maddpg.agents], os.path.join(res_dir, 'model.pt'))
np.save(os.path.join(res_dir, 'reward.npy'), total_reward)
        
def get_running_reward(reward_array: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(reward_array)
    for i in range(window - 1):
        running_reward[i] = np.mean(reward_array[:i + 1])
    for i in range(window - 1, len(reward_array)):
        running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
    return running_reward

# plot result
fig, ax = plt.subplots()
x = range(1, EPISODES + 1)
for agent in range(NUM_GOOD + NUM_ADVERSARIES):
    ax.plot(x, total_reward[:, agent], label=agent)
    ax.plot(x, get_running_reward(total_reward[:, agent]))
ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('reward')
title = f'training result of maddpg solve {ENV_NAME}'
ax.set_title(title)
plt.savefig(os.path.join(res_dir, title))
env.close()