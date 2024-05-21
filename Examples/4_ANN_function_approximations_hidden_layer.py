# Artificial Neural Networks as Universal Function Approximators Part II


# In the previous section, we discussed artificial neural networks as function 
# approximators, with the intention of using them to replace the tables used to store 
# Q-values in algorithms such as Q-learning and SARSA. We explored the concept of linear 
# separability, which refers to the ability to separate data or points by a line, plane, or 
# hyperplane when we represent them in the space of their coordinates, i.e., the space 
# defined by axes corresponding to each feature (corresponding to a position in the vector 
# of the datapoint). With a single layer of 
# neurons, we are limited to learning problems that are linearly separable. To tackle more 
# complex problems, we need several layers as well as a non-linear activation function. 
# These modifications introduce the necessary non-linearity that allows the network to model 
# complex patterns in data that single-layer networks cannot, thus broadening the scope of 
# challenges the network can learn from and solve.

# The simplest non-linearly separable problem
# XOR is the simplest non-linearly separable problem because it requires two lines to separate the classes 0 and 1. A layer of neurons will be unable to learn it.

import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Plot XOR data points
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', label='Class 1')

# Labeling and formatting
plt.title('XOR Problem')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1.15))
plt.grid(True)
plt.show()

# Exercise
# You have the task of learning the function of a noisy OR function, using a single layer of neurons. You can play with the script; report your best performance.
# The code uses the He initialization to initialize the weights of the network

# Define original XOR function inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

# Generate noisy data around the original XOR function outputs to form clusters
np.random.seed(42)  # For reproducibility
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the noisy XOR data points in 2D
plt.figure(figsize=(8, 6))
plt.scatter(noisy_inputs[:, 0], noisy_inputs[:, 1], c=noisy_outputs, cmap='coolwarm', s=20, alpha=0.8)
plt.colorbar(label='Output (Noisy)')
plt.title('2D Plot of Noisy XOR Problem')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.grid(True)
plt.show()

# Solution

# Generate XOR data with noise
def generate_noisy_or_data(samples, noise_level=0.1):
    x = np.random.randint(2, size=(samples, 2))
    y = np.bitwise_xor(x[:, 0], x[:, 1]).reshape(-1, 1)
    x_noisy = x + np.random.normal(0, noise_level, x.shape)
    y_noisy = y + np.random.normal(0, noise_level, y.shape)
    return x_noisy, y_noisy

class SingleLayerModel:
    def __init__(self, input_dim, output_dim):
        # He initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((output_dim,))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        h = np.dot(X, self.W) + self.b
        y = self.sigmoid(h)
        return y

    def train(self, X, y_t, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            error = y_pred - y_t
            delta = error * y_pred * (1 - y_pred)
            dW = np.dot(X.T, delta) / X.shape[0]
            db = np.sum(delta, axis=0) / X.shape[0]

            self.W -= learning_rate * dW
            self.b -= learning_rate * db
            
            if epoch % 100 == 0:
                loss = np.mean(0.5 * (error ** 2))
                print(f"Epoch {epoch}, Loss: {loss}")

# Generate data
training_data = generate_noisy_or_data(2000)
validation_data = generate_noisy_or_data(300)

# Instantiate and train the model
model = SingleLayerModel(2, 1)  # 2 inputs, 1 output
model.train(training_data[0], training_data[1], learning_rate=0.1, epochs=2000)

# Evaluate the model's performance
def evaluate_model(model, data):
    X_test, y_test = data
    predictions = model.forward(X_test)
    decisions = predictions > 0.5
    accuracy = np.mean(decisions == np.round(y_test))
    return accuracy

validation_accuracy = evaluate_model(model, validation_data)
print(f"Model Validation Accuracy: {validation_accuracy}")


# Derivation of the Backpropagation Algorithm for the Hidden Layers with Sigmoid Activation

# Exercise
# In the previous example of modelling the noisy XOR add a hidden layer. What is the performance now?

# Solution

# Function to generate XOR data with noise
def generate_noisy_or_data(samples, noise_level=0.1):
    x = np.random.randint(2, size=(samples, 2))
    y = np.bitwise_xor(x[:, 0], x[:, 1]).reshape(-1, 1)
    x_noisy = x + np.random.normal(0, noise_level, x.shape)
    y_noisy = y + np.random.normal(0, noise_level, y.shape)
    return x_noisy, y_noisy

class MultiLayerModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((output_dim,))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.sigmoid(self.z2)

    def train(self, X, y_t, learning_rate, epochs):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_t_shuffled = y_t[indices]

            # Forward pass
            y_pred = self.forward(X_shuffled)
            error = y_pred - y_t_shuffled

            # Backpropagation
            delta2 = error * y_pred * (1 - y_pred)
            dW2 = np.dot(self.a1.T, delta2) / n_samples
            db2 = np.sum(delta2, axis=0) / n_samples
            delta1 = np.dot(delta2, self.W2.T) * (self.z1 > 0)
            dW1 = np.dot(X_shuffled.T, delta1) / n_samples
            db1 = np.sum(delta1, axis=0) / n_samples

            # Update weights and biases
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            if epoch % 100 == 0:
                loss = np.mean(0.5 * (error ** 2))
                print(f"Epoch {epoch}, Loss: {loss}")

# Generate datasets
training_data = generate_noisy_or_data(3000)
validation_data = generate_noisy_or_data(300)
test_data = generate_noisy_or_data(300)

# Instantiate and train the model
model = MultiLayerModel(2, 4, 1)  # 2 inputs, 4 hidden neurons, 1 output
model.train(training_data[0], training_data[1], learning_rate=0.1, epochs=2000)

# Evaluate the model's performance
def evaluate_model(model, data):
    X_test, y_test = data
    predictions = model.forward(X_test)
    decisions = predictions > 0.5
    accuracy = np.mean(decisions == np.round(y_test))
    return accuracy

# Evaluation
validation_accuracy = evaluate_model(model, validation_data)
print(f"Model Validation Accuracy: {validation_accuracy}")
test_accuracy = evaluate_model(model, test_data)
print(f"Model Test Accuracy: {test_accuracy}")

# Replacing the Q-values Table with an ANN in Q-learning

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class LinearTrackQAgent:
    def __init__(self, track_size, epsilon=0.05, eta=0.1, gamma=0.99):
        self.states = track_size
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.actions = 2
        self.W = np.random.randn(self.states, self.actions) * 0.01
        self.b = np.zeros(self.actions,)

    def choose_action(self, state):
        pass
        
    def update_Q(self, state, action, reward, next_state):
        pass


    def train(self, episodes, runs=100, max_steps=100):
        pass

    def plot_learning_progress(self, average_steps_per_episode, optimal_steps_per_episode):
        plt.plot(average_steps_per_episode, label='Average Steps per Episode')
        plt.axhline(y=optimal_steps_per_episode, color='r', linestyle='--', label='Optimal Steps')
        plt.title('Learning Progress over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.legend()
        plt.show()

# Initialize the agent with a track size of 6 states
agent = LinearTrackQAgent(track_size=6,epsilon=0.01, eta=0.01, gamma=0.9)
optimal_steps_per_episode = 5

# Train the agent over 100 episodes and 10 runs, then plot the learning progress

# Solution

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class LinearTrackQAgent:
    def __init__(self, track_size, epsilon=0.05, eta=0.1, gamma=0.99):
        self.states = track_size
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.actions = 2
        self.W = np.random.randn(self.states, self.actions) * 0.01
        self.b = np.zeros(self.actions,)

    def choose_action(self, state):
        X = np.zeros((1,self.states))
        X[0,state] = 1
        h=np.matmul(X,self.W)+ self.b
        q_values = sigmoid(h)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(q_values)

    def update_Q(self, state, action, reward, next_state):
        X = np.zeros((1,self.states))
        X[0,state] = 1
        next_state_X= np.zeros((1,self.states))
        next_state_X[0,next_state] = 1
        
        # Compute the Q values for current and next state
        h=np.matmul(X, self.W) + self.b
        q_values = sigmoid(h)
        next_q_values = sigmoid(np.dot(next_state_X, self.W) + self.b)
        
        # Compute the TD target and error
        target = reward + self.gamma * np.max(next_q_values)

        adjusted_target = np.copy(q_values)
        # Update the target only for the taken actions across the batch
        adjusted_target[0, action] = target
        error =  (q_values -adjusted_target) * sigmoid_derivative(h) 
 
        # Update gradient calculation to use the correct action index

        grad_W = np.matmul(X.T, error) / X.shape[0]
        grad_b = np.sum(error, axis=0) / X.shape[0]
        
        self.W -= self.eta * grad_W
        self.b -= self.eta * grad_b


    def train(self, episodes, runs=100, max_steps=100):
        steps_run_episode = np.zeros((runs, episodes))
        for run in range(runs):
            for episode in range(episodes):
                state = 0
                steps = 0
                while state < self.states - 1 and steps < max_steps:
                    action = self.choose_action(state)
                    next_state = state + 1 if action == 1 else max(0, state - 1)
                    reward = 1 if next_state == self.states - 1 else 0
                    self.update_Q(state, action, reward, next_state)
                    state = next_state
                    steps += 1
                steps_run_episode[run, episode] = steps
        return steps_run_episode

    def plot_learning_progress(self, average_steps_per_episode, optimal_steps_per_episode):
        plt.plot(average_steps_per_episode, label='Average Steps per Episode')
        plt.axhline(y=optimal_steps_per_episode, color='r', linestyle='--', label='Optimal Steps')
        plt.title('Learning Progress over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.legend()
        plt.show()

# Initialize the agent with a track size of 6 states

states=20

agent = LinearTrackQAgent(track_size=states,epsilon=0.01, eta=0.01, gamma=0.99)
optimal_steps_per_episode = states-1

# Train the agent over 100 episodes and 10 runs, then plot the learning progress
steps_run_episode = agent.train(episodes=100, runs=50, max_steps=1000)
average_steps_per_episode = np.mean(steps_run_episode, axis=0)
agent.plot_learning_progress(average_steps_per_episode, optimal_steps_per_episode)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class LinearTrackQAgent:
    def __init__(self, track_size, hidden_size=10, epsilon=0.05, eta=0.1, gamma=0.99):
        self.states = track_size
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.actions = 2
        self.W1 = np.random.randn(self.states, self.hidden_size) * 0.01
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.actions) * 0.01
        self.b2 = np.zeros(self.actions)

    def choose_action(self, state):
        X = np.zeros((1, self.states))
        X[0, state] = 1
        hidden = sigmoid(np.dot(X, self.W1) + self.b1)
        q_values = sigmoid(np.dot(hidden, self.W2) + self.b2)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(q_values)

    def update_Q(self, state, action, reward, next_state):
        X = np.zeros((1, self.states))
        X[0, state] = 1
        next_state_X = np.zeros((1, self.states))
        next_state_X[0, next_state] = 1

        hidden = sigmoid(np.dot(X, self.W1) + self.b1)
        q_values = sigmoid(np.dot(hidden, self.W2) + self.b2)

        next_hidden = sigmoid(np.dot(next_state_X, self.W1) + self.b1)
        next_q_values = sigmoid(np.dot(next_hidden, self.W2) + self.b2)

        target = reward + self.gamma * np.max(next_q_values)
        adjusted_target = np.copy(q_values)
        adjusted_target[0, action] = target

        # Compute the error terms
        delta_output = (adjusted_target - q_values) * sigmoid_derivative(np.dot(hidden, self.W2) + self.b2)
        delta_hidden = np.dot(delta_output, self.W2.T) * sigmoid_derivative(np.dot(X, self.W1) + self.b1)

        # Compute gradients
        grad_W2 = np.dot(hidden.T, delta_output)
        grad_b2 = np.sum(delta_output, axis=0)
        grad_W1 = np.dot(X.T, delta_hidden)
        grad_b1 = np.sum(delta_hidden, axis=0)

        # Update weights and biases
        self.W2 += self.eta * grad_W2
        self.b2 += self.eta * grad_b2
        self.W1 += self.eta * grad_W1
        self.b1 += self.eta * grad_b1

    def train(self, episodes, runs=100, max_steps=100):
        steps_run_episode = np.zeros((runs, episodes))
        for run in range(runs):
            for episode in range(episodes):
                state = 0
                steps = 0
                while state < self.states - 1 and steps < max_steps:
                    action = self.choose_action(state)
                    next_state = state + 1 if action == 1 else max(0, state - 1)
                    reward = 1 if next_state == self.states - 1 else 0
                    self.update_Q(state, action, reward, next_state)
                    state = next_state
                    steps += 1
                steps_run_episode[run, episode] = steps
        return steps_run_episode

    def plot_learning_progress(self, average_steps_per_episode, optimal_steps_per_episode):
        plt.plot(average_steps_per_episode, label='Average Steps per Episode')
        plt.axhline(y=optimal_steps_per_episode, color='r', linestyle='--', label='Optimal Steps')
        plt.title('Learning Progress over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Number of Steps')
        plt.legend()
        plt.show()

# Parameters for initialization and training
states = 25
agent = LinearTrackQAgent(track_size=states, hidden_size=10, epsilon=0.05, eta=0.1, gamma=0.99)
optimal_steps_per_episode = states - 1

# Train the agent and plot the learning progress
steps_run_episode = agent.train(episodes=200, runs=50, max_steps=1000)
average_steps_per_episode = np.mean(steps_run_episode, axis=0)
agent.plot_learning_progress(average_steps_per_episode, optimal_steps_per_episode)