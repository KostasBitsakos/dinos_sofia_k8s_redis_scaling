import numpy as np
import pandas as pd
from collections import deque
import random
from network import MyDenseNet  # Import the new dense network
import torch
import torch.nn as nn
from environment import Environment

# Define the Double DQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.online_model = MyDenseNet(state_size, 128, action_size)
        self.target_model = MyDenseNet(state_size, 128, action_size)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Use Huber Loss

        self.loss_history = []

    def save_model(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, verbose=False):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32)

        self.online_model.eval()
        with torch.no_grad():
            act_values = self.online_model.forward(state)

        return torch.argmax(act_values).item()

    def plot_loss_history(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.savefig('losses.jpg')
        plt.close('all')

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        batch_losses = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = (reward + self.gamma * torch.max(self.target_model.forward(next_state)))

            state = torch.tensor(state, dtype=torch.float32)

            self.online_model.eval()
            with torch.no_grad():
                target_f = self.online_model.forward(state)
                target_f[action] = target

            self.online_model.train()
            self.online_model.zero_grad()
            output = self.online_model.forward(state)
            loss = self.criterion(output, target_f.detach())
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()
        self.loss_history.append(np.mean(batch_losses))

if __name__ == "__main__":
    agent = DDQNAgent(6, 3)
    env = Environment()
    actions = [0, 1, 2]  # Possible actions

    for _ in range(10):  # Run for 10 steps
        action = np.random.choice(actions)  # Random action
        state, reward, load_reward, latency_penalty, vm_penalty, cpu_penalty, memory_penalty, done = env.step(action)
        print(f"Time: {env.time}")
        print(f"Load: {state[0]}, Capacity: {state[1]}, Reward: {reward}")
        print(f"CPU Usage: {state[2]}, Memory Usage: {state[3]}, Latency: {state[4]}")
        print(f"Number of VMs: {state[5]}\n")
