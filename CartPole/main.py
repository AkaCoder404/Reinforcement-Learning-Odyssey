"""
Title: RL for Classic Control CartPole

Description:
Actions (2)
    - 0: Push to the left
    - 1: Push to the right
    
Observations (4)
"""

import gymnasium as gym
from q_learn import QLearn
import numpy as np
from gymnasium.utils.save_video import save_video

# ENVIRONMENT
EPISODES = 50000
LEARNING_RATE = 0.1
DROPOUT_FACTOR = 0.95


EPSILON = 0.20
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

# Initialize the environment
env = gym.make('CartPole-v1')

num_states = env.observation_space.shape[0] # 4
num_actions = env.action_space.n            # 2

cart_position_high, cart_position_low = env.observation_space.high[0], env.observation_space.low[0]
cart_velocity_high, cart_velocity_low = env.observation_space.high[1], env.observation_space.low[1]
pole_angle_high, pole_angle_low = env.observation_space.high[2], env.observation_space.low[2]
pole_velocity_high, pole_velocity_low = env.observation_space.high[3], env.observation_space.low[3]

print(f"Number of states: {num_states}")
print(f"Number of actions: {num_actions}")
print(f"Cart Position High: {cart_position_high}, Cart Position Low: {cart_position_low}")
print(f"Cart Velocity High: {cart_velocity_high}, Cart Velocity Low: {cart_velocity_low}")
print(f"Pole Angle High: {pole_angle_high}, Pole Angle Low: {pole_angle_low}")
print(f"Pole Velocity High: {pole_velocity_high}, Pole Velocity Low: {pole_velocity_low}")

# Initialize Q-table
Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
shape = Observation + [env.action_space.n]
qlearn = QLearn(shape, EPSILON, DROPOUT_FACTOR, LEARNING_RATE)
print("Qlearn qtable", qlearn.get_q_table().shape)

def get_discrete_state(state):
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(int))

# Train the agent (Q-learning)
print("Start Episodes")
total_rewards = []
for episode in range(EPISODES):
    observation, info = env.reset()
    discrete_state = get_discrete_state(observation)
    # print(f"Observation {observation}")
    # print(f"Discrete state {discrete_state}")
    
    termination, truncation = False, False
    episode_rewards = 0
    while not termination and not truncation:
        # Declare action
        action = qlearn.select_action(discrete_state, env.action_space)
        next_observation, reward, termination, truncation, info = env.step(action)
        next_discrete_state = get_discrete_state(next_observation)
        
        # Q-learning update
        qlearn.learn(discrete_state, next_discrete_state, action, reward)
        discrete_state = next_discrete_state
        episode_rewards += reward
        
    total_rewards.append(episode_rewards)
    

env.close()

# Save plot of total_rewards per episode
print("Plotting total rewards...")
import matplotlib.pyplot as plt
plt.plot(total_rewards)
# Plot rolling average
window_size = 1000
rolling_average = np.convolve(total_rewards, np.ones(window_size), "valid") / window_size
plt.plot(rolling_average)
plt.savefig("total_rewards.png")

# Evaluate the trained Q-table
num_evaluation_episodes = 10
env = gym.make('CartPole-v1', render_mode="rgb_array")

total_rewards = []
for eval_eps in range(num_evaluation_episodes):
    observation, info = env.reset()
    termination, truncation = False, False
    discrete_state = get_discrete_state(observation)
    table = qlearn.get_q_table()

    frames = []
    episode_rewards = 0
    while not termination and not truncation:
        # Render and save frame
        frames.append(env.render())
        action = qlearn.select_action(discrete_state, env.action_space)
        observation, reward, termination, truncation, info = env.step(action)    
        discrete_state = get_discrete_state(observation)
        episode_rewards += reward
    
    total_rewards.append(episode_rewards)
        
    # Save video
    print("Frame Length: ", len(frames))
    if episode_rewards <= 250:
        continue
    
    if frames:
        save_video(
            frames,
            video_folder="videos",
            fps=env.metadata["render_fps"],
            name_prefix="cartpole",
            # step_starting_index=step,
            # episode_index=episode
        )
    env.reset()


average_reward = np.mean(total_rewards)
print(f"Average Reward over {num_evaluation_episodes} Evaluation Episodes: {average_reward:.2f}")
env.close()