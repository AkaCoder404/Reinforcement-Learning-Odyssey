import numpy as np

class QLearn():
    def __init__(self, shape, epsilon, discount_factor, learning_rate):
        # Initialize Q-table
        self.q_table = np.zeros(shape)
        
        self.q_table = np.random.uniform(low=0, high=1, size=shape)
        
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
    # Function for epsilon-greedy action selection
    def select_action(self, state, action_space):
        if np.random.rand() < self.epsilon:
            return action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
        
    
    # Update Q-table
    def learn(self, curr_state, next_state, action, reward):
        self.q_table[curr_state + (action,)] = \
            self.q_table[curr_state + (action,)] + \
            self.learning_rate * \
            (reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[curr_state + (action,)])

    # Get Q-table
    def get_q_table(self):
        return self.q_table