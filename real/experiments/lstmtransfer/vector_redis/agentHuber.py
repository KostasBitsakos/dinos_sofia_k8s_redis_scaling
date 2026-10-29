import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from networks import myLSTM

class DDQNAgentLSTM:
    def __init__(self, state_size, action_size, sequence_length, pretrained_model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1  
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00005  
        self.online_model = myLSTM(state_size, 128, 3, action_size)
        self.target_model = myLSTM(state_size, 128, 3, action_size)
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        self.loss_history = []

        # Save the pretrained model path for future use
        self.pretrained_model = pretrained_model

        if pretrained_model:
            print(f"🔄 Loading pre-trained model from: {pretrained_model}")
            self.online_model.load_state_dict(torch.load(pretrained_model))
            self.target_model.load_state_dict(torch.load(pretrained_model))
            self.epsilon = 0.2  # Increase exploration slightly for better evaluation
            print("✅ Pre-trained model loaded successfully!")


    def save_model(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, (1, self.sequence_length, self.state_size))
        state = torch.tensor(state, dtype=torch.float32)

        self.online_model.eval()
        with torch.no_grad():
            act_values = self.online_model.forward(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        batch_losses = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, (1, self.sequence_length, self.state_size))
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.target_model.forward(next_state))

            state = np.reshape(state, (1, self.sequence_length, self.state_size))
            state = torch.tensor(state, dtype=torch.float32)

            self.online_model.eval()
            with torch.no_grad():
                target_f = self.online_model.forward(state)
                target_f[0][action] = target

            self.online_model.train()
            self.optimizer.zero_grad()
            output = self.online_model.forward(state)
            loss = self.criterion(output, target_f.detach())
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()
        self.loss_history.append(np.mean(batch_losses))

    def fine_tune(self, env, fine_tune_steps=1000, batch_size=16):
        state = np.zeros((self.sequence_length, self.state_size))
        for step in range(fine_tune_steps):
            action = self.act(state)
            next_state, reward, _, _, _, _, _, done = env.step(action)

            next_state_sequence = np.zeros((self.sequence_length, self.state_size))
            next_state_sequence[:-1] = state[1:]
            next_state_sequence[-1] = next_state

            self.remember(state, action, reward, next_state_sequence, done)
            if len(self.memory) > batch_size:
                self.replay(batch_size)

            state = next_state_sequence
            if done:
                print(f"⚠️ Fine-tuning stopped early at step {step} due to 'done' condition.")
                break

        print(f"✅ Fine-tuning complete for {fine_tune_steps} steps.")
