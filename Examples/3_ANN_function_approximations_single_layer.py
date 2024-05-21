# Artificial Neural Networks as Universal Function Approximators

# Exercise: Robot in a Linear Track
 
# A robot is placed at the beginning of a linear track (state 0) and must navigate to a reward at the end of the track (state 5). The robot has two possible actions, "forward" and "backward". The track consists of 6 discrete states, numbered from 0 to 5. The objective is to use the Q-learning reinforcement learning algorithm to teach the robot to reach the reward in an optimal manner, which means taking the minimum number of steps.

# Objectives:  

# 1. In the following template, implement the Q-learning algorithm.
# 2. Identify the minimum number of required steps to reach the reward. 
# 3. Identify the minimum number of episodes required for the robot to learn how to reach the reward using the minimum number of steps. 
# 4. Monitor and plot the number of steps the robot takes to reach the goal over episodes to observe a learning curve.
# 5. Expand the track to 30 states; which is now the optimal number of steps to reach the reward?
# 6. What is the total number of Q-values for the expanded track?
# 7. What is the necessary number of episodes for learning to reach the reward with the minimum number of steps in this extended track?
#  8. Plot the Q-value of the forward action as a function of the state - you can do this for one run.
# 9. Is there a difference between the number of epochs that the algorithm requires to correctly learn the reward values and the number of epochs to learn to go to the goal with the optimal number of steps

# Goals:

# The aim of the exercise is to revise key concepts by demonstrating that:
# 1. Correct learning reflects an optimal or near-optimal solution regarding the number of actions needed to reach one's goal.
# 2. The number of Q-values to be learned changes with a more difficult problem, and so does the time to learn.
# 3. Q-values are a function of the state-action pair, but we will isolate one action (forward in this case) to demonstrate this concept in one dimension.


# Note:
# We have highlighted the need for statistics. You will need to repeat the episodes for ten runs and observe average plots. You may plot the Q-values of one run or the average Q-values across runs.

import numpy as np
import matplotlib.pyplot as plt

class LinearTrackEnvironment:
    def __init__(self, track_size):
        self.track_size = track_size
        self.actions=2
    
    def reset(self):
        self.state=0
        self.terminal=0
        return self.state, self.terminal

    def step(self, state, action):
        if action == 1:  # Move forward
            next_state = min(state + 1, self.track_size - 1)
        else:  # Move backward
            next_state = max(0, state - 1)
        reward = 1 if next_state == self.track_size - 1 else 0
        self.terminal= 1 if next_state == self.track_size - 1 else 0
        return next_state, reward, self.terminal

class LinearTrackAgent:
    def __init__(self, environment, epsilon=0.05, eta=0.1, gamma=0.99):
        self.env = environment
        self.epsilon = epsilon  # Exploration rate
        self.eta = eta      # Learning rate
        self.gamma = gamma  # Discount factor
        self.Q = np.zeros((self.env.track_size, self.env.actions))  # Initialize Q-values for each state and action (forward, backward)
        self.max_steps=0  #Initialise max numbers of steps allowed
        self.episodes=0   #Initialise number of episodes
        self.steps_per_episode = []  # Track the number of steps per episode

    def choose_action(self, state):
        pass

    def update_Q(self, state, action, reward, next_state):
        self.Q[state, action] += self.eta * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def train(self, episodes=20, runs=100, max_steps=100):
        self.max_steps=max_steps
        self.episodes=episodes
        steps_run_episode = np.zeros((runs, episodes))  # Corrected the shape of the array
        for run in range(runs):
            self.Q = np.zeros((self.env.track_size, self.env.actions))  # Reinitialize Q-values for each run
            self.steps_per_episode = []  # Reset steps per episode for each run
            for episode in range(episodes):
                state, terminal = env.reset()       
                steps = 0  # Reset step count for the episode
                while not terminal and steps < self.max_steps:
                    # TO DO
                    pass
                steps_run_episode[run, episode] = steps
        return steps_run_episode, self.Q 
    
    def plot_learning_progress(self, average_steps_per_episode, optimal_steps_per_episode):
          # TO DO
        plt.show()

# Initialize the agent with a track size of 6 states
states=6

env = LinearTrackEnvironment(track_size=6)
agent = LinearTrackAgent(environment=env)

# Train the agent over 20 episodes and 50 runs, then plot the learning progress

# Solution

class LinearTrackEnvironment:
    def __init__(self, track_size):
        self.track_size = track_size
        self.actions=2
    
    def reset(self):
        self.state=0
        self.terminal=0
        return self.state, self.terminal

    def step(self, state, action):
        if action == 1:  # Move forward
            next_state = min(state + 1, self.track_size - 1)
        else:  # Move backward
            next_state = max(0, state - 1)
        reward = 1 if next_state == self.track_size - 1 else 0
        self.terminal= 1 if next_state == self.track_size - 1 else 0
        return next_state, reward, self.terminal

class LinearTrackAgent:
    def __init__(self, environment, epsilon=0.05, eta=0.1, gamma=0.99):
        self.env = environment
        self.epsilon = epsilon  # Exploration rate
        self.eta = eta      # Learning rate
        self.gamma = gamma  # Discount factor
        self.Q = np.zeros((self.env.track_size, self.env.actions))  # Initialize Q-values for each state and action (forward, backward)
        self.max_steps=0  #Initialise max numbers of steps allowed
        self.episodes=0   #Initialise number of episodes
        self.steps_per_episode = []  # Track the number of steps per episode

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action= np.random.choice([0, 1])  # Explore
        else:
            # Handle case where Q-values are the same by randomly choosing among the best actions
            max_q = np.max(self.Q[state])
            action=np.random.choice(np.where(self.Q[state] == max_q)[0])
        return action

    def update_Q(self, state, action, reward, next_state):
        self.Q[state, action] += self.eta * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def train(self, episodes=20, runs=100, max_steps=100):
        self.max_steps=max_steps
        self.episodes=episodes
        steps_run_episode = np.zeros((runs, episodes))  # Corrected the shape of the array
        for run in range(runs):
            self.Q = np.zeros((self.env.track_size, self.env.actions))  # Reinitialize Q-values for each run
            self.steps_per_episode = []  # Reset steps per episode for each run
            for episode in range(episodes):
                state, terminal = env.reset()       
                steps = 0  # Reset step count for the episode
                while not terminal and steps < self.max_steps:
                    action = self.choose_action(state)
                    next_state, reward, terminal = self.env.step(state, action)
                    self.update_Q(state, action, reward, next_state)
                    state = next_state
                    steps += 1  # Increment step count
                steps_run_episode[run, episode] = steps
        return steps_run_episode, self.Q

    def plot_learning_progress(self, average_steps_per_episode, optimal_steps_per_episode):
        plt.plot(average_steps_per_episode, label='Average Steps per Episode')
        plt.axhline(y=optimal_steps_per_episode, color='r', linestyle='--', label='Optimal Steps')
        plt.title('Learning Progress over Episodes - Track Size: ' + str(self.env.track_size))
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.xlim(0, self.episodes)  # Set the y-axis limits
        plt.ylim(0, self.max_steps)  # Set the y-axis limits
        plt.legend()
        plt.show()

# Initialize the agent with a track size of 6 states
states=6
epsilon=0.01
eta=0.1
gamma=0.99
optimal_steps_per_episode=states-1
max_steps=100 #if you increase the number of states you may also ineed to increase the number of steps to learn
episodes=10
runs=50

env = LinearTrackEnvironment(track_size=states)
agent = LinearTrackAgent(environment=env, epsilon=epsilon, eta=eta, gamma=gamma)

# Train the agent over 100 episodes and 10 runs, then plot the learning progress
steps_run_episode, Q_values = agent.train(episodes=episodes, runs=runs, max_steps=max_steps)
average_steps_per_episode = np.mean(steps_run_episode, axis=0)  # Calculate the average across runs
agent.plot_learning_progress(average_steps_per_episode,optimal_steps_per_episode)

Q_values

for row in Q_values:
    print('[', end='')
    for value in row:
        print("{:.3f}".format(value), end=', ')
    print(']')

# Approximating the Q-value Function with an Artificial Neural Network
# Solutions
# Define original OR function inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the inputs with outputs as the z-axis 
ax.scatter(inputs[:, 0], inputs[:, 1], outputs, c=outputs, cmap='coolwarm', s=200, marker='x', linewidth=3)

# Draw a cube to encapsulate the points for clearer visualization
# Define the edges of the cube
edges = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0],
         [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1],
         [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]]
edges = np.array(edges)

# Plot the edges of the cube
for start, end in zip(edges, edges[1:]):
    ax.plot3D(*zip(start, end), color="black", linestyle='--')

# Add the green hyperplane for separability
xx, yy = np.meshgrid(range(2), range(2))
zz = 0.5 * np.ones_like(xx)  # Define the plane at z = 0.5

# Plot the separating hyperplane
ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

# Label axes
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
ax.set_title('3D Plot of OR Function with Cube and Separating Hyperplane')

# Set ticks for z-axis
ax.set_zticks([0, 1])

plt.show()

# Exercise: Noisy OR function:

# Define original OR function inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 1])

# Generate noisy data around the original OR function outputs to form clusters
np.random.seed(42)  # For reproducibility

# Define the number of noisy points per original point
num_noisy_points = 100

# Initialize arrays to hold noisy inputs and outputs
noisy_inputs = []
noisy_outputs = []

# Generate noisy data around each of the original OR function points
for input_point, output_point in zip(inputs, outputs):
    # Generate noisy inputs around the current point
    noisy_inputs.append(input_point + np.random.normal(0, 0.1, size=(num_noisy_points, 2)))
    # Generate noisy outputs around the current output (0 or 1)
    noisy_outputs.append(np.ones(num_noisy_points) * output_point)

# Convert the lists of arrays into single numpy arrays
noisy_inputs = np.concatenate(noisy_inputs)
noisy_outputs = np.concatenate(noisy_outputs)

# Recreate the 3D plot with the noisy data and include the cube for better visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the noisy data
ax.scatter(noisy_inputs[:, 0], noisy_inputs[:, 1], noisy_outputs, c=noisy_outputs, cmap='coolwarm', s=20, depthshade=False)

# Draw a cube to encapsulate the points for clearer visualization
edges = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0],
         [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1],
         [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]]
edges = np.array(edges)

# Plot the edges of the cube
for start, end in zip(edges, edges[1:]):
    ax.plot3D(*zip(start, end), color="black", linestyle='--')

# Add the green hyperplane for separability
xx, yy = np.meshgrid(range(2), range(2))
zz = 0.5 * np.ones_like(xx)  # Define the plane at z = 0.5 for visualization

ax.set_box_aspect([1,1,0.6])  #  aspect ratio

# Plot the separating hyperplane
ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

# Label axes
ax.set_xlabel('Input 1', labelpad=10)
ax.set_ylabel('Input 2', labelpad=10)
ax.set_zlabel('Output (Noisy)', labelpad=10)
ax.set_title('3D Plot of OR Function with Noisy Clusters and Cube')

# Set ticks for z-axis
plt.tight_layout()
plt.show()

# Task 1: Noisy OR Data Generation
def generate_noisy_or_data(samples, noise_level=0.1):
    # Generate binary inputs
    x = np.random.randint(2, size=(samples, 2))
    # OR function
    y = np.bitwise_or(x[:, 0], x[:, 1]).reshape(-1, 1)
    # Add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_level, x.shape)
    y_noisy = y + np.random.normal(0, noise_level, y.shape)
    return x_noisy, y_noisy


# Generate datasets
training_data = generate_noisy_or_data(1000)
validation_data = generate_noisy_or_data(200)
test_data = generate_noisy_or_data(200)

# Define the SingleNeuronModel class with np.matmul for matrix multiplication
class SingleNeuronModel:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((output_dim,))
    
    def activation(self, h):
        return 1 / (1 + np.exp(-h))
    
    def output(self, X):
        # Use np.matmul for matrix multiplication
        pass
    
    def train(self, X, y_t, learning_rate, epochs):
        for epoch in range(epochs):
            pass
        pass
          

# Evaluate the model's performance
def evaluate_model(model, data):
    pass
     

# Instantiate and train the model
model = SingleNeuronModel(2, 1)  # OR function: 2 inputs, 1 output
model.train(training_data[0], training_data[1], learning_rate=0.1, epochs=1000)

# Evaluate the revised model
validation_accuracy = evaluate_model(model, test_data)
print(" Model Validation Accuracy: {validation_accuracy_}")

# Evaluate the revised model - you only do this after finishing with your model
#test_accuracy = evaluate_model(model, test_data)
#print(" Model Test Accuracy: {test_accuracy_}")

# Task 1: Noisy OR Data Generation
def generate_noisy_or_data(samples, noise_level=0.1):
    # Generate binary inputs
    x = np.random.randint(2, size=(samples, 2))
    # OR function
    y = np.bitwise_or(x[:, 0], x[:, 1]).reshape(-1, 1)
    # Add Gaussian noise
    x_noisy = x + np.random.normal(0, noise_level, x.shape)
    y_noisy = y + np.random.normal(0, noise_level, y.shape)
    return x_noisy, y_noisy

# Generate datasets
training_data = generate_noisy_or_data(2000)
validation_data = generate_noisy_or_data(300)
test_data = generate_noisy_or_data(300)

# Define the SingleNeuronModel class with np.matmul for matrix multiplication
class SingleNeuronModel:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((output_dim,))
    
    def activation(self, h):
        return 1 / (1 + np.exp(-h))
    
    def output(self, X):
        # Use np.matmul for matrix multiplication
        linear_output = np.matmul(X, self.W) + self.b
        return self.activation(linear_output)
    
    def train(self, X, y_t, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.output(X)
            delta = y_pred - y_t
            dW = np.matmul(X.T, delta * y_pred * (1 - y_pred)) / X.shape[0]
            db = np.sum(delta * y_pred * (1 - y_pred), axis=0) / X.shape[0]
            self.W -= learning_rate * dW
            self.b -= learning_rate * db
            if epoch % 100 == 0:
                error = np.mean(0.5 * (delta ** 2))
                print(f"Epoch {epoch}, Loss: {error}")

# Evaluate the model's performance
def evaluate_model(model, data):
    X_test, y_test = data
    predictions = model.output(X_test)
    decisions = predictions > 0.5
    accuracy = np.mean(decisions == np.round(y_test))
    return accuracy

print(training_data[0].shape)
print(training_data[1].shape)

# Instantiate and train the model
model = SingleNeuronModel(2, 1)  # OR function: 2 inputs, 1 output
model.train(training_data[0], training_data[1], learning_rate=0.1, epochs=1000)

# Use to optimise the model 
validation_accuracy = evaluate_model(model, validation_data)
print(f" Model Validation Accuracy: {validation_accuracy}")

# Evaluate the  model
test_accuracy = evaluate_model(model, test_data)
print(f" Model Test Accuracy: {test_accuracy}")

import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Step 1: Data Loading
def load_boston_data(url):
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode('utf-8').split('\n')
    data = np.genfromtxt(lines, delimiter=',', skip_header=1)
    X = data[:, 5].reshape(-1, 1)  # average number of rooms (RM)
    y = data[:, -1]  # median value of owner-occupied homes (MEDV)
    return X, y

url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'  # URL to the Boston Housing dataset
X, y = load_boston_data(url)

plt.scatter(X, y, label='Actual Prices')
plt.xlabel('Average Number of Rooms')
plt.ylabel('House Prices ($1000s)')
plt.title('Boston Housing')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Splitting the dataset
def split_data(X, y, train_size=0.7, valid_size=0.15, test_size=0.15, shuffle=True, seed=None):
    if shuffle:
        np.random.seed(seed)
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    num_samples = len(X)
    num_train = int(train_size * num_samples)
    num_valid = int(valid_size * num_samples)
    
    X_train, y_train = X[:num_train], y[:num_train]
    X_valid, y_valid = X[num_train:num_train+num_valid], y[num_train:num_train+num_valid]
    X_test, y_test = X[num_train+num_valid:], y[num_train+num_valid:]
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Step 3: Splitting the dataset with shuffling
X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y, train_size=0.7, valid_size=0.15, test_size=0.15, shuffle=True, seed=42)

# Printing the shapes of the splits
print("Train set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_valid.shape, y_valid.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting training data
axs[0].scatter(X_train, y_train, label='Training Data', color='blue')
axs[0].set_title('Training Data')
axs[0].set_xlabel('Average Number of Rooms')
axs[0].set_ylabel('House Prices ($1000s)')
axs[0].legend()
axs[0].grid(True)

# Plotting validation data
axs[1].scatter(X_valid, y_valid, label='Validation Data', color='green')
axs[1].set_title('Validation Data')
axs[1].set_xlabel('Average Number of Rooms')
axs[1].set_ylabel('House Prices ($1000s)')
axs[1].legend()
axs[1].grid(True)

# Plotting test data
axs[2].scatter(X_test, y_test, label='Test Data', color='red')
axs[2].set_title('Test Data')
axs[2].set_xlabel('Average Number of Rooms')
axs[2].set_ylabel('House Prices ($1000s)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Step 1: Data Loading
def load_boston_data(url):
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode('utf-8').split('\n')
    data = np.genfromtxt(lines, delimiter=',', skip_header=1)
    X = data[:, 5].reshape(-1, 1)  # average number of rooms (RM)
    y = data[:, -1]  # median value of owner-occupied homes (MEDV)
    return X, y

url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'  # URL to the Boston Housing dataset
X, y = load_boston_data(url)

plt.scatter(X, y, label='Actual Prices')
plt.xlabel('Average Number of Rooms')
plt.ylabel('House Prices ($1000s)')
plt.title('Boston Housing')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Splitting the dataset
def split_data(X, y, train_size=0.7, valid_size=0.15, test_size=0.15, shuffle=True, seed=None):
    if shuffle:
        np.random.seed(seed)
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    num_samples = len(X)
    num_train = int(train_size * num_samples)
    num_valid = int(valid_size * num_samples)
    
    X_train, y_train = X[:num_train], y[:num_train]
    X_valid, y_valid = X[num_train:num_train+num_valid], y[num_train:num_train+num_valid]
    X_test, y_test = X[num_train+num_valid:], y[num_train+num_valid:]
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Step 3: Splitting the dataset with shuffling
X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y, train_size=0.7, valid_size=0.15, test_size=0.15, shuffle=True, seed=42)

# Printing the shapes of the splits
print("Train set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_valid.shape, y_valid.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Define the SingleNeuronModel class with np.matmul for matrix multiplication
class SingleNeuronModel:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) 
        self.b = np.zeros((output_dim,))
    
    def relu(self, h):
       return np.maximum(0, h)
 
    def relu_derivative(self, h):
       return np.where(h > 0, 1, 0)
    
    def linear_output(self, X):
        # Use np.matmul for matrix multiplication
        linear_output = np.matmul(X, self.W) + self.b
        return linear_output

    def predict(self,X):
         return self.relu(self.linear_output(X))
    
    def train(self, X, y_t, learning_rate, epochs):
        for epoch in range(epochs):
            h=self.linear_output(X)
            y_pred = self.relu(h)
            delta = y_pred - y_t
            dW = np.matmul(X.T, delta * self.relu_derivative(h))  / X.shape[0]
            db = np.sum(delta * self.relu_derivative(h), axis=0) / X.shape[0]
            self.W -= learning_rate * dW
            self.b -= learning_rate * db
            if epoch % 1000 == 0:
                error = np.mean(0.5 * (delta ** 2))
                print(f"Epoch {epoch}, Error: {error}")


print(y_train.shape)

# # Instantiate and train the model
model = SingleNeuronModel(1, 1)  # OR function: 2 inputs, 1 output

# # Train the model on the entire dataset
model.train(X_train, y_train, learning_rate=0.01, epochs=10000)

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plotting training data
axs[0].scatter(X_train, y_train, label='Training Data', color='blue')
axs[0].set_title('Training Data')
axs[0].set_xlabel('Average Number of Rooms')
axs[0].set_ylabel('House Prices ($1000s)')
axs[0].legend()
axs[0].grid(True)

# Plotting validation data
axs[1].scatter(X_valid, y_valid, label='Validation Data', color='green')
axs[1].set_title('Validation Data')
axs[1].set_xlabel('Average Number of Rooms')
axs[1].set_ylabel('House Prices ($1000s)')
axs[1].legend()
axs[1].grid(True)

# Plotting test data
axs[2].scatter(X_test, y_test, label='Test Data', color='red')
axs[2].set_title('Test Data')
axs[2].set_xlabel('Average Number of Rooms')
axs[2].set_ylabel('House Prices ($1000s)')
axs[2].legend()
axs[2].grid(True)

# Plotting the model
# Plot the final model on the training data
axs[0].plot(X_train, model.predict(X_train), color='orange', linewidth=2, label='Final Model')
# Plot the final model on the validation data
axs[1].plot(X_valid, model.predict(X_valid), color='orange', linewidth=2, label='Final Model')
# Plot the final model on the test data
axs[2].plot(X_test, model.predict(X_test), color='orange', linewidth=2, label='Final Model')


plt.tight_layout()
plt.show()