import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from networks import myLSTM  # Assuming myLSTM is defined in networks.py

class DDQNAgent:
    def __init__(self, state_size, action_size, sequence_length):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=5000)  # Increased memory size for stability
        self.gamma = 0.98  # Discount factor to prioritize long-term rewards
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.05  # Minimum exploration to avoid full exploitation
        self.epsilon_decay = 0.995  # Slow decay for smoother transition from exploration
        self.learning_rate = 0.0001  # Slightly reduced for more stable training
        
        # Define the LSTM models for online and target networks
        self.online_model = myLSTM(state_size, 128, 3, action_size)
        self.target_model = myLSTM(state_size, 128, 3, action_size)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss for robustness against outliers
        self.loss_history = []  # Track loss during training

    def save_model(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, verbose=False):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.tensor(state.reshape(1, self.sequence_length, self.state_size), dtype=torch.float32)
        self.online_model.eval()
        with torch.no_grad():
            action_values = self.online_model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        """Train the model using past experiences."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        batch_losses = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state.reshape(1, self.sequence_length, self.state_size), dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor))

            state_tensor = torch.tensor(state.reshape(1, self.sequence_length, self.state_size), dtype=torch.float32)
            self.online_model.eval()
            with torch.no_grad():
                target_f = self.online_model(state_tensor)
            target_f[0][action] = target

            self.online_model.train()
            self.optimizer.zero_grad()
            output = self.online_model(state_tensor)
            loss = self.criterion(output, target_f.detach())
            loss.backward()
            self.optimizer.step()
            batch_losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()
        self.loss_history.append(np.mean(batch_losses))

    def plot_loss_history(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history)
        plt.title("Loss Over Time")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig('final_metrics_original/losses.jpg')
        plt.close()
