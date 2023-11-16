"""
Title: RL for ToyText FrozenLake

Observations - represnets the current state of the environment (where the agent is) 4x4 grid
"""

enum_action = { 0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP" }


import gymnasium as gym
from gymnasium.utils.save_video import save_video
import numpy as np
import time

from q_learn import QLearn

# ENVIRONMENT
EPISODES =  10000
LEARNING_RATE = 1
DISCOUNT_FACTOR = 0.90
EPSILON = 0.90
MAP = "8x8"
ENVIRONMENT_NAME = "FrozenLake-v1"


# Initialize the environment
env = gym.make(ENVIRONMENT_NAME, map_name=MAP, is_slippery=False)
observation, info = env.reset()

print("Observatons", env.observation_space, env.observation_space.sample())

# Initialize Q-table
print("Initialize Q-table...")
num_states = env.observation_space.n


num_actions = env.action_space.n
print("Number of states", num_states)
print("Number of actions", num_actions)
q_table = np.zeros((num_states, num_actions))
print("Q-table", q_table.shape)

qlearn = QLearn(num_states, num_actions, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE)

total_rewards = []
for step in range(EPISODES):
    observation, info = env.reset()
    print("Observations", observation, info)
    terminated = False
    truncated = False
    
    episode_rewards = 0
    while not terminated and not truncated:
        # Declare action
        action = qlearn.select_action(observation, env.action_space)
        next_observation, reward, terminated, truncated, info = env.step(action)
        # Q-learning update
        qlearn.learn(observation, next_observation, action, reward)
        episode_rewards += reward
        observation = next_observation  
        
    print(f"Episode {step}: Reward: {reward}")
    total_rewards.append(episode_rewards)
env.close()

# Save plot of total_rewards per episode
print("Plotting total rewards...")
import matplotlib.pyplot as plt
plt.plot(total_rewards)
# Plot rolling average
window_size = 100
rolling_average = np.convolve(total_rewards, np.ones(window_size), "valid") / window_size
plt.savefig("total_rewards.png")


# Evaluate the trained Q-table
print("Evaluating the trained Q-table...")
num_evaluation_episodes = 1
total_rewards = []
# Reinitalize the environment
env = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array", map_name=MAP, is_slippery=False)
for episode in range(num_evaluation_episodes):
    observation, info = env.reset()
    state = observation
    total_reward = 0
    terminated = False
    truncated = False
    q_table = qlearn.get_q_table()
    # print(q_table)

    step = 0
    frames = [] # Save frame for video
    print(env.metadata)
    while not terminated and not truncated:
        # Render and save frame
        frames.append(env.render())

        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
    
    print(f"Episode {episode}... Total Reward: {total_reward}")
    total_rewards.append(total_reward)
   

    # Save video
    print("Frame Length: ", len(frames))
    if total_reward < 150:
        continue
    
    if frames:
        save_video(
            frames,
            video_folder="videos",
            fps=env.metadata["render_fps"],
            name_prefix=MAP
            # step_starting_index=step,
            # episode_index=episode
        )
    env.reset()

    if total_rewards:
        average_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_evaluation_episodes} Evaluation Episodes: {average_reward:.2f}")
    else:
        print("No episodes were evaluated.")
        
env.close()
print(f"Average Reward over {num_evaluation_episodes} Evaluation Episodes: {average_reward:.2f}")
