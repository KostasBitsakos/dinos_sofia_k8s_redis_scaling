import numpy as np
import os
import torch
from environment_dense import Environment
from network import MyDenseNet  # Use the dense network
import torch.nn as nn

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.online_model = MyDenseNet(state_size, 128, action_size)
        self.target_model = MyDenseNet(state_size, 128, action_size)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def save_model(self, filename):
        torch.save(self.online_model.state_dict(), filename)

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        self.online_model.eval()
        with torch.no_grad():
            action_values = self.online_model(state)
        return torch.argmax(action_values).item()

    def fine_tune(self, pretrained_model_path, fine_tune_steps=1000, batch_size=16, output_folder="new_metrics"):
        self.online_model.load_state_dict(torch.load(pretrained_model_path))
        env = Environment()
        state = np.zeros(self.state_size)

        os.makedirs(output_folder, exist_ok=True)
        loss_file = os.path.join(output_folder, f"losses_{pretrained_model_path}.txt")

        for step in range(fine_tune_steps):
            action = self.act(state)
            next_state, reward, _, _, _, _, _, done = env.step(action)

            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) > batch_size:
                minibatch = random.sample(self.memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        target += self.gamma * torch.max(self.target_model(torch.tensor(s_next, dtype=torch.float32)))

                    self.online_model.zero_grad()
                    target_f = self.online_model(torch.tensor(s, dtype=torch.float32))
                    loss = self.criterion(target_f[a], torch.tensor(target))
                    loss.backward()
                    self.optimizer.step()

            with open(loss_file, "a") as f:
                f.write(f"{loss.item()}\n")

            state = next_state
            if done:
                break

        fine_tuned_model_path = pretrained_model_path.replace(".pth", "_fine_tuned.pth")
        self.save_model(fine_tuned_model_path)
