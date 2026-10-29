import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environment import Environment  # Import the Environment class

# DDQN Agent class definition
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
    # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
    # Save the model's state dictionary
        torch.save(self.model.state_dict(), name)

    def load(self, name):
    # Load the model's state dictionary
        self.model.load_state_dict(torch.load(name))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
                return np.argmax(action_values.cpu().data.numpy())
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
        # Convert to tensors
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        # Get the current Q values for all actions
            current_q_values = self.model(state_tensor)

        # Compute the target Q value
            target_q_value = reward
            if not done:
                target_q_value = (reward + self.gamma * torch.max(self.target_model(next_state_tensor).detach()).item())

        # Update the target Q value for the action taken
            current_q_values[0][action] = target_q_value

        # Perform a gradient descent step
            self.optimizer.zero_grad()
            loss = nn.functional.mse_loss(current_q_values, torch.tensor([target_q_value]).float())
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
# Training function
def train_agent(env, agent, episodes=2000, steps_per_episode=500):
    batch_size = 32
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        for time in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
        if e % 10 == 0:
            agent.update_target_model()
            agent.save('final_model.pth')

# Evaluation function
def evaluate_agent(env, agent, episodes=2000, steps_per_episode=500):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        for time in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = np.reshape(next_state, [1, agent.state_size])
            if done:
                break
        print(f"Evaluation Episode: {e+1}/{episodes}, Total Reward: {total_reward}")


# Main function
def main():
    env = Environment()
    state_size = 5  # Define the size of the state
    action_size = 3  # Define the size of the action space
    agent = DDQNAgent(state_size, action_size)

    print("Training the agent...")
    train_agent(env, agent)

    print("Evaluating the agent...")
    evaluate_agent(env, agent)

    # Save the final model
    agent.save('final_model.pth')

if __name__ == "__main__":
    main()
