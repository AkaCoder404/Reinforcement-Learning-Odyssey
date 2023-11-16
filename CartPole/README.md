# Cart Pole

## Cart Pole with Q-Learning


## Continuous to Discrete State
The `Box` class in OpenAI Gym is a space that represents an n-dimensional box. This can be used when you have **multi-dimensional continuous space**. The four dimensions correspond to the following observation.

1. Cart Position: Ranges from -4.8 to 4.8
2. Cart Velocity: Can be any float.
3. Pole Angle: Ranges from -24 degrees to 24 degrees
4. Pole Velocity At Tip: Can be any float.

A continuous space means an infinite possible of states. The function `get_discrete_state` function is used to discretize the state space, which means we divide the continuous state space into a finite number of discrete space.

In the function, the continuous state is divided by `np_array_win_size` which can be thought of as the size of each discrete state, and then the offset is added `np.array([15, 10, 1, 10])`. The result is then conerted to an integer and returned as a tuple, which represents the discrete state.

> This however, also also introduces some approximation error, as the discrete states are only an approximation of true continuous states. This specific discritization method is known as `uniform discretization`.

The choice of `[30, 30, 50, 50]` for the discretization of the state space in the CartPole environment is somewhat arbitrary and is likely based on trial and error. The goal is to find a balance between having enough states to accurately represent the environment, but not so many that learning becomes slow or infeasible.

In the code, the `Observation` variable `[30, 30, 50, 50]` represents the discretized state space. Each number corresponds to the number of discrete states for each observation in the environment's observation space. The CartPole environment has four observations (Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip), and so we've chosen to discretize each into 30, 30, 50, and 50 discrete states respectively. The `env.action_space.n` is the number of possible actions that the agent can take. In the CartPole environment, there are two possible actions: move the cart to the left or to the right.

So, the Q-table is of size `[30, 30, 50, 50, 2]` because for each combination of discrete states (30 options for the first observation, 30 for the second, 50 for the third, and 50 for the fourth), there are 2 possible actions. The Q-table stores the expected reward for each action in each of these states.

*So why was 30, 30, 50, 50 chosen?* Here's a breakdown:

1. Cart Position: This value is relatively less critical to the balance of the pole, so a lower number of bins (30) is chosen.
2. Cart Velocity: Similar to Cart Position, the velocity of the cart is less critical, so a lower number of bins (30) is chosen.
3. Pole Angle: The angle of the pole is very critical to keeping the pole balanced. A small change in angle can result in the pole falling over. Therefore, a higher number of bins (50) is chosen to more accurately represent this value.
4. Pole Velocity At Tip: Similar to Pole Angle, the velocity of the pole tip is very critical to the balance of the pole. Therefore, a higher number of bins (50) is chosen.

Remember, these values are not set in stone. They are hyperparameters of the model and can be tuned to optimize performance. Different problems and different environments may require different levels of discretization.

*So why does lower number of bins mean less importance?* The number of bins in discretization is directly related to the level of detail or precision with which you represent the state space.

When you have a larger number of bins, you are dividing the continuous state space into a larger number of discrete states. This means that you can represent the state space with more precision, as each bin corresponds to a smaller range of the continuous state space. This can be beneficial when small changes in the state can have a large impact on the outcome, which is why more bins are used for more critical parts of the state space.

On the other hand, when you have a smaller number of bins, you are dividing the continuous state space into a smaller number of discrete states. Each bin corresponds to a larger range of the continuous state space, so you are representing the state space with less precision. This can be sufficient when small changes in the state have less impact on the outcome, which is why fewer bins are used for less critical parts of the state space.

However, it's important to note that while using more bins can increase precision, it also increases the size of the Q-table and thus the complexity of the learning problem. Therefore, it's a trade-off between precision and computational complexity.

*So what about np_array_win_size?*
The `np_array_win_size` is used to determine the size of the bins when discretizing the continuous state space. It is chosen based on the range of values each observation can take in the CartPole environment.

The `np_array_win_size` is calculated by taking the difference between the high and low values of the observation space and dividing by the number of bins for each observation. This gives the width of each bin, which is used in the get_discrete_state function to convert a continuous state into a discrete state.

