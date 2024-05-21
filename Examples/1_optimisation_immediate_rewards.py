# Immediate Rewards

# Introduction to Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic function f(x) = ax^2 + bx + c
def f(x, a, b, c):
    return a*x**2 + b*x + c

# Coefficients
a, b, c = 1, -4, 3

# Generate x values
x = np.linspace(-10, 10, 400)

# Calculate y values
y = f(x, a, b, c)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='$f(x)$')
plt.title('Plot of the Convex Function $f(x) = ax^2 + bx + c$, a>0')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True)
plt.show()

# Multiple Local Minima
# Define a function with two local minima and its derivative
def f(x):
    return x**4 - 2*x**3 - 12*x**2 + 2

def df(x):
    return 4*x**3 - 6*x**2 - 24*x

# Implement gradient descent
def gradient_descent(starting_point, learning_rate, num_iterations):
    x = starting_point
    history = [x]
    for _ in range(num_iterations):
        grad = df(x)
        x -= learning_rate * grad
        history.append(x)
    return np.array(history), f(np.array(history))

# Randomize the starting point within a specific range to explore different minima
starting_point = np.random.uniform(-4, 4)
learning_rate = 0.01
num_iterations = 50

history, function_values = gradient_descent(starting_point, learning_rate, num_iterations)

# Plotting
x_values = np.linspace(-4, 4, 400)
y_values = f(x_values)
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="Function with Two Local Minima")
plt.scatter(history, function_values, color='red', zorder=5, label="Gradient Descent Steps")
plt.plot(history, function_values, color='red', linestyle='dashed', zorder=5)
# Add arrow for initial position
plt.annotate('Initial Position', xy=(history[0], function_values[0]), xytext=(history[0]+1, function_values[0]+50),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.title("Function with Two Local Minima and Gradient Descent")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

class BanditProblem:
    def __init__(self, k=10):
        self.k = k
        # Probabilities of getting a reward for each arm
        self.p_success = np.random.uniform(0.1, 0.9, self.k)
        # Assign reward magnitudes inversely related to their probability
        # Higher rewards for arms with lower success probabilities
        self.reward_magnitude = (1 / self.p_success -1)

    def get_reward(self, action):
        # Binary reward based on the arm's success probability
        success = np.random.rand() < self.p_success[action]
        return success * self.reward_magnitude[action]

    def return_expectations(self):
        # Expected rewards for each arm
        return self.p_success * self.reward_magnitude

    def return_arms(self):
        #Returns number of arms
        return self.k

    def plot_reward_frequency(self):
        plt.figure(figsize=(10, 6))
        
        # Calculate expected rewards for each arm
        reward_frequencies = self.p_success
        
        # Create a bar plot of expected rewards
        arms = np.arange(1, self.k + 1)  # Arm indices for x-axis
        plt.bar(arms, reward_frequencies, color='blue')
        
        plt.title('Reward Frequency for Each Arm')
        plt.xlabel('Arm')
        plt.ylabel('Reward Frequency')
        plt.xticks(arms)  # Ensure a tick for each arm
        plt.savefig('binary_bandit2.png')  # Save the figure as a PNG file
        plt.show()

    def plot_true_reward_distributions(self):
        plt.figure(figsize=(10, 6))
        
        # Calculate expected rewards for each arm
        expected_rewards = self.return_expectations()
        
        # Create a bar plot of expected rewards
        arms = np.arange(1, self.k + 1)  # Arm indices for x-axis
        plt.bar(arms, expected_rewards, color='skyblue')
        
        plt.title('Expected Reward for Each Arm')
        plt.xlabel('Arm')
        plt.ylabel('Expected Reward')
        plt.xticks(arms)  # Ensure a tick for each arm
        plt.savefig('binary_bandit.png')  # Save the figure as a PNG file
        plt.show()


class Policy:
    def __init__(self, number_of_actions, eta=None):
        self.number_of_actions = number_of_actions
        self.eta=eta
        self.reset()
        # Choose the update method based on whether eta is provided
        if self.eta is None:
            self.update = self.update_average
        else:
            self.update = self.update_eta

    def reset(self):
        self.Q = np.zeros(self.number_of_actions)  # Reset estimated rewards
        self.action_counts = np.zeros(self.number_of_actions)  # Reset counts of selections

    def update_average(self, action, reward):
        # TODO:implement stationary environment update rule
        pass
    
    def update_eta(self, action, reward):
        # TODO: implement online update rule
        pass

    def update_estimates(self, action, reward):
        # This method will now delegate to the correct update method chosen at initialization
        self.update(action, reward)

    def select_action(self):
        raise NotImplementedError("Subclasses should implement this method.")
         
        
class EpsilonGreedyPolicy(Policy):
    def __init__(self, number_of_actions, eta, epsilon):
        super().__init__(number_of_actions, eta)
        self.epsilon = epsilon

    def select_action(self):
         #TODO: implement policy
        pass

class SoftMaxPolicy(Policy):
    def __init__(self, number_of_actions, eta, tau):
        super().__init__(number_of_actions, eta)
        self.tau = tau

    def select_action(self):
        #TODO: implement policy
        pass


def simulate_bandit_policy(policy, bandit_problem, trials=100):
    reward_history = np.zeros(trials)
    policy.reset()  # Ensure the strategy is reset at the beginning of each simulation
    for trial in range(trials):
        #TODO: select action
        #TODO: get reward fromt bandit
        #TODO: update policy
        reward_history[trial] = reward
    return reward_history

def smooth_data(data, alpha=0.1):
    if len(data) == 0:  # Check for empty data
        return np.array([])  # Return an empty array if data is empty
    
    smoothed_data = np.zeros(len(data))  # Initialize smoothed_data with zeros
    smoothed_data[0] = data[0]  # First data point remains the same

    for i in range(1, len(data)):  # Start loop from the second element
        smoothed_data[i] = (1 - alpha) * smoothed_data[i - 1] + alpha * data[i]

    return smoothed_data

def simulate_and_average_policy(policy, bandit_problem, steps=100, runs=10):
   
    sum_of_rewards = np.zeros(steps)
    for _ in range(runs):
        rewards = simulate_bandit_policy(policy, bandit_problem, steps)
        sum_of_rewards += rewards
    return smooth_data(sum_of_rewards / runs, 0.01)


k = 10
trials = 10000
episodes = 100
epsilon = 0.12
tau=0.15
eta=None

# Initialize the bandit problem  
my_bandit = BanditProblem(k)

my_bandit.plot_true_reward_distributions()

class BanditProblem:
    def __init__(self, k=10):
        self.k = k
        # Probabilities of getting a reward for each arm
        self.p_success = np.random.uniform(0.1, 0.9, self.k)
        # Assign reward magnitudes inversely related to their probability
        # Higher rewards for arms with lower success probabilities
        self.reward_magnitude = (1 / self.p_success -1)

    def get_reward(self, action):
        # Binary reward based on the arm's success probability
        success = np.random.rand() < self.p_success[action]
        return success * self.reward_magnitude[action]

    def return_expectations(self):
        # Expected rewards for each arm
        return self.p_success * self.reward_magnitude

    def return_arms(self):
        #Returns number of arms
        return self.k

    def plot_reward_frequency(self):
        plt.figure(figsize=(10, 6))
        
        # Calculate expected rewards for each arm
        reward_frequencies = self.p_success
        
        # Create a bar plot of expected rewards
        arms = np.arange(1, self.k + 1)  # Arm indices for x-axis
        plt.bar(arms, reward_frequencies, color='blue')
        
        plt.title('Reward Frequency for Each Arm')
        plt.xlabel('Arm')
        plt.ylabel('Reward Frequency')
        plt.xticks(arms)  # Ensure a tick for each arm
        plt.savefig('binary_bandit2.png')  # Save the figure as a PNG file
        plt.show()

    def plot_true_reward_distributions(self):
        plt.figure(figsize=(10, 6))
        
        # Calculate expected rewards for each arm
        expected_rewards = self.return_expectations()
        
        # Create a bar plot of expected rewards
        arms = np.arange(1, self.k + 1)  # Arm indices for x-axis
        plt.bar(arms, expected_rewards, color='skyblue')
        
        plt.title('Expected Reward for Each Arm')
        plt.xlabel('Arm')
        plt.ylabel('Expected Reward')
        plt.xticks(arms)  # Ensure a tick for each arm
        plt.savefig('binary_bandit.png')  # Save the figure as a PNG file
        plt.show()

class Policy:
    def __init__(self, number_of_actions, eta=None):
        self.number_of_actions = number_of_actions
        self.eta=eta
        self.reset()
        # Choose the update method based on whether eta is provided
        if self.eta is None:
            self.update = self.update_average
        else:
            self.update = self.update_eta

    def reset(self):
        self.Q = np.zeros(self.number_of_actions)  # Reset estimated rewards
        self.action_counts = np.zeros(self.number_of_actions)  # Reset counts of selections


    def update_average(self, action, reward):
        self.action_counts[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.action_counts[action] 
    
    def update_eta(self, action, reward):
        self.Q[action] += self.eta * (reward - self.Q[action])

    def update_estimates(self, action, reward):
        # This method will now delegate to the correct update method chosen at initialization
        self.update(action, reward)

    def select_action(self):
        raise NotImplementedError("Subclasses should implement this method.")


class EpsilonGreedyPolicy(Policy):
    def __init__(self, number_of_actions, eta, epsilon):
        super().__init__(number_of_actions, eta)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:  # Exploration
            return np.random.randint(0, self.number_of_actions)
        else:  # Exploitation
            max_q_value = np.max(self.Q)
            actions_with_max_q = np.where(self.Q == max_q_value)[0]
            return np.random.choice(actions_with_max_q)
        
class SoftMaxPolicy(Policy):
    def __init__(self, number_of_actions, eta, tau):
        super().__init__(number_of_actions, eta)
        self.tau = tau

    def select_action(self):
        exp_Q = np.exp(self.Q / self.tau)
        probabilities = exp_Q / np.sum(exp_Q)
        return np.random.choice(range(self.number_of_actions), p=probabilities)


def simulate_bandit_policy(policy, bandit_problem, trials=100):
    reward_history = np.zeros(trials)
    policy.reset()  # Ensure the strategy is reset at the beginning of each simulation
    for trial in range(trials):
        action = policy.select_action()
        reward = bandit_problem.get_reward(action)  # Fetch reward from the bandit problem
        policy.update_estimates(action, reward)  # Update strategy with the observed reward
        reward_history[trial] = reward
    return reward_history

def smooth_data(data, alpha=0.1):
    if len(data) == 0:  # Check for empty data
        return np.array([])  # Return an empty array if data is empty
    
    smoothed_data = np.zeros(len(data))  # Initialize smoothed_data with zeros
    smoothed_data[0] = data[0]  # First data point remains the same

    for i in range(1, len(data)):  # Start loop from the second element
        smoothed_data[i] = (1 - alpha) * smoothed_data[i - 1] + alpha * data[i]

    return smoothed_data

def simulate_and_average_policy(policy, bandit_problem, trials=100, episodes=10):
    sum_of_rewards = np.zeros(trials)
    for _ in range(episodes):
        rewards = simulate_bandit_policy(policy, bandit_problem, trials)
        sum_of_rewards += rewards
    return smooth_data(sum_of_rewards / episodes, 0.01)


k = 10
trials = 10000
episodes = 100
epsilon = 0.2
tau=0.3
eta=0.001

# Initialize the bandit problem and strategies
#np.random.seed(10) #for testing
my_bandit = BanditProblem(k)

my_bandit.plot_true_reward_distributions()
my_bandit.plot_reward_frequency()

epsilon_greedy = EpsilonGreedyPolicy(my_bandit.return_arms(), eta, epsilon)
greedy = EpsilonGreedyPolicy(my_bandit.return_arms(), eta, 0)
soft_max=SoftMaxPolicy(my_bandit.return_arms(), eta, tau)

# Simulate and average across runs

rewards_epsilon_greedy=simulate_and_average_policy(epsilon_greedy, my_bandit, trials, episodes)
rewards_greedy=simulate_and_average_policy(greedy, my_bandit, trials, episodes)
rewards_soft_max=simulate_and_average_policy(soft_max, my_bandit, trials, episodes)

# Display the true mean rewards for each arm
print("True mean rewards for each arm: \n", my_bandit.return_expectations())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(rewards_epsilon_greedy, label="Epsilon-Greedy")
plt.plot(rewards_greedy, label="Greedy")
plt.plot(rewards_soft_max, label="Soft-Max")
max_expected_reward = np.max(my_bandit.return_expectations())
mean_expected_reward = np.mean(my_bandit.return_expectations())
plt.axhline(y=max_expected_reward, color='r', linestyle='--', label=f'Max Expected Reward: {max_expected_reward:.2f}')
plt.axhline(y=mean_expected_reward, color='b', linestyle='--', label=f'Mean Expected Reward: {mean_expected_reward:.2f}')
plt.xlabel('Trials')
plt.ylabel('Average Reward across Episodes')
plt.title('Performance of various Policies Over Time')
plt.legend()
plt.savefig('binary_bandit_performance.png')  # Save the figure as a PNG file
plt.show()