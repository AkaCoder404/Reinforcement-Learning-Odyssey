import numpy as np
import torch
import torch.nn as nn

class QLearn():
    def __init__(self, num_states, num_actions, epsilon, discount_factor, learning_rate):
        self.q_table = np.zeros((num_states, num_actions))
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
    # Function for epsilon-greedy action selection
    def select_action(self, state, action_space):
        if np.random.rand() < self.epsilon:
            return action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit
        
    
    # Update Q-table
    def learn(self, curr_state, next_state, action, reward):
        self.q_table[curr_state, action] = \
            self.q_table[curr_state, action] + \
            self.learning_rate * \
            (reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[curr_state, action])

    # Get Q-table
    def get_q_table(self):
        return self.q_table
    
# Agent class
class DQNAgent():
    def __init__(self, num_states, num_actions, capacity, learning_rate, epsilon):
        """ Initialize the DQN
        :param num_states: Number of states
        :param num_actions: Number of actions
        :param capacity: Capacity of the buffer
        
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.loss = nn.MSELoss()
        
        # Main model, fit on every step, gets trained on every step
        self.model = MLP(self.num_states, self.num_actions)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Target model, predict on every step
        self.target_model = MLP(self.num_states, self.num_actions)
        self.model_optimizer = torch.optim.Adam(self.target_model.parameters(), lr=self.learning_rate)
        
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Buffer
        self.buffer = []
                
    def select_action(self, state, action_space):
        if np.random.rand() < self.epsilon:
            return action_space.sample()
        else:
            return np.argmax(self.q_table[state, :])
        
        
class MLP(nn.Module):
    def __init__(self, num_states, num_actions):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(self.num_states, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.num_actions)
        
        # Sequential
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )     
    
    def forward(self, x):
        return self.model(x)
        