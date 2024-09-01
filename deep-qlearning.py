import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 10
NUM_EPISODES = 500


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample() 
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item() 


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    
    experiences = memory.sample(BATCH_SIZE)
    batch = list(zip(*experiences))

    states = torch.tensor(batch[0], dtype=torch.float32)
    actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(batch[3], dtype=torch.float32)
    dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

  
    current_q_values = policy_net(states).gather(1, actions)

    
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)

   
    target_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

    
    loss = nn.MSELoss()(current_q_values, target_q_values)

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = gym.make('CartPole-v1')
policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net = DQN(env.observation_space.shape[0], env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

epsilon = EPSILON_START
rewards_per_episode = []


for episode in range(NUM_EPISODES):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, policy_net, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        optimize_model()

    rewards_per_episode.append(total_reward)

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 10 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}')

env.close()

# Plot the rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Performance on CartPole-v1')
plt.show()
