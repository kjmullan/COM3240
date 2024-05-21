# Future Rewards II: Temporal Difference Learning

# Exercise

# An agent operates in a 12 x 4 environment. Its starting position is located at coordinates $(3, 0)$, aiming to reach the goal at the opposite end at $(3, 11)$. This scenario introduces a "cliff" spanning from $(3, 1)$ to $(3, 10)$. Any step into this zone results in the agent being sent back to the start, receiving a large negative reward of $-100$. The environment supports four actions: up, down, left, and right, represented numerically as $0, 1, 2,$ and $3$, respectively. The `step` function dictates the transition dynamics, including the penalties for falling off the cliff and the conditions for staying within the grid boundaries.

# This scenario is inspired by an example in the book by Sutton and Barto on reinforcement learning. The `train` function encapsulates the learning process, employing an `epsilon_greedy_policy` to balance exploration and exploitation. The choice between SARSA and Q-Learning is determined by the `method` parameter.

# Your task is to implement the missing parts in the code provided below and observe how they differ in addressing this problem. Discuss why this is the case.

# Tasks:
# 1. Implement the `epsilon_greedy_policy` function to choose actions based on the current state and Q-values.
# 2. Complete the sections marked `TODO` for calculating the target for both SARSA and Q-Learning algorithms.
# 3. Update the Q-values based on the calculated targets.
# 4. Run the training process for both SARSA and Q-Learning methods.
# 5. Plot the average rewards per episode to compare the performance of the two methods.
# 6. Discuss the observed differences in how SARSA and Q-Learning handle the "cliff" environment. Consider why one method might perform better or differently than the other in this specific scenario.

# Ensure your discussion includes insights into the exploration-exploitation trade-off, the impact of immediate vs. future rewards, and how each algorithm's update mechanism influences its behavior in the presence of high-penalty states like the cliff.

import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnv:
    def __init__(self):
        self.width = 12
        self.height = 4
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, state, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
        if next_state in self.cliff:
            return self.start, -100, True
        if next_state == self.goal:
            return next_state, 0, True
        if 0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width:
            return next_state, -1, False
        return state, -1, False

# TODO: Implement the epsilon_greedy_policy function
def epsilon_greedy_policy(Q, state, epsilon=0.1):
    # Implement the epsilon-greedy policy here
    pass

def train(env, episodes, eta, gamma, epsilon, method):
    Q = np.zeros((env.height, env.width, 4))
    episode_rewards = []


    for _ in range(episodes):
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            # TODO: Use epsilon_greedy_policy to select an action
            action = None  # Placeholder for action selection
            next_state, reward, done = env.step(state, action)
            
            if method == 'sarsa':
                # TODO: Select next_action using epsilon_greedy_policy for SARSA
                next_action = None  # Placeholder for next action selection in SARSA
                # Target for SARSA: Q-value of the next state-action pair
                target = None  # TODO: Calculate the target for SARSA
            elif method == 'q_learning':
                # Target for Q-Learning: max Q-value of the next state across all possible actions
                target = None  # TODO: Calculate the target for Q-Learning
            
            # TODO: Update the Q-value for the current state and action using the target
            # Placeholder for Q-value update

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    return episode_rewards

env = CliffWalkingEnv()
episodes = 1000
eta = 0.1
gamma = 0.999
epsilon = 0.1

class CliffWalkingEnv:
    def __init__(self):
        self.width = 12
        self.height = 4
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, state, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
        if next_state in self.cliff:
            return self.start, -100, True
        if next_state == self.goal:
            return next_state, 0, True
        if 0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width:
            return next_state, -1, False
        return state, -1, False

def epsilon_greedy_policy(Q, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        max_value = np.max(Q[state])
        best_actions = np.flatnonzero(Q[state] == max_value)
        return np.random.choice(best_actions)

def train(env, episodes, eta, gamma, epsilon, method):
    Q = np.zeros((env.height, env.width, 4))
    episode_rewards = []

    for _ in range(episodes):
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(state, action)
            if method == 'sarsa':
                next_action = epsilon_greedy_policy(Q, next_state, epsilon)
                target = Q[next_state][next_action]
            elif method == 'q_learning':
                target = np.max(Q[next_state])
            Q[state][action] += eta * (reward + gamma * target - Q[state][action])
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    return episode_rewards

env = CliffWalkingEnv()
episodes = 1000
eta = 0.1
gamma = 0.999
epsilon = 0.1

sarsa_rewards = train(env, episodes, eta, gamma, epsilon, 'sarsa')
q_learning_rewards = train(env, episodes, eta, gamma, epsilon, 'q_learning')

# Plotting
plt.plot(np.mean(np.array(sarsa_rewards).reshape(-1, 10), axis=1), label='SARSA')
plt.plot(np.mean(np.array(q_learning_rewards).reshape(-1, 10), axis=1), label='Q-Learning')
plt.xlabel('Episodes (grouped in tens)')
plt.ylabel('Average Reward')
plt.legend()
plt.title('SARSA vs Q-Learning')
plt.show()
