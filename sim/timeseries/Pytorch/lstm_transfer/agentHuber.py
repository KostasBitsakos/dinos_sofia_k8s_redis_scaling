import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from networks import myLSTM

class DDQNAgentLSTM:
    def __init__(self, state_size, action_size, sequence_length, pretrained_model=None,
                 legacy=False, target_update_every=100):
        # legacy=True reproduces the published behaviour (plain DQN target, target synced on
        # every replay, carried LSTM hidden state, per-sample optimizer steps).
        self.legacy = legacy
        self.target_update_every = target_update_every  # in replay() calls
        self.replay_calls = 0
        self.optimizer_steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.memory = deque(maxlen=5000)
        self.gamma = 0.98  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1  
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00005  
        self.online_model = myLSTM(state_size, 128, 3, action_size, legacy=legacy)
        self.target_model = myLSTM(state_size, 128, 3, action_size, legacy=legacy)
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
        if self.legacy:
            return self._replay_legacy(batch_size)
        minibatch = random.sample(self.memory, batch_size)
        shape = (batch_size, self.sequence_length, self.state_size)
        states = torch.tensor(np.array([m[0] for m in minibatch]).reshape(shape), dtype=torch.float32)
        actions = torch.tensor([m[1] for m in minibatch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32)
        next_states = torch.tensor(np.array([m[3] for m in minibatch]).reshape(shape), dtype=torch.float32)
        dones = torch.tensor([float(m[4]) for m in minibatch], dtype=torch.float32)

        targets = self.compute_targets(rewards, next_states, dones)

        self.online_model.train()
        q = self.online_model(states).gather(1, actions).squeeze(1)
        loss = self.criterion(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_steps += 1

        self.replay_calls += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.replay_calls % self.target_update_every == 0:
            self.update_target_model()
        self.loss_history.append(loss.item())

    def compute_targets(self, rewards, next_states, dones):
        """Double DQN: a* = argmax_a Q_online(s'), y = r + gamma (1-done) Q_target(s', a*)."""
        with torch.no_grad():
            a_star = self.online_model(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_model(next_states).gather(1, a_star).squeeze(1)
            return rewards + self.gamma * (1.0 - dones) * q_next

    def _replay_legacy(self, batch_size):
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
            self.optimizer_steps += 1

            batch_losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.replay_calls += 1
        self.update_target_model()
        self.loss_history.append(np.mean(batch_losses))

    def fine_tune(self, env, fine_tune_steps=1000, batch_size=16):
        if not self.legacy:
            env.reset()  # F2: a finished env (time>=5000) would end fine-tuning at step 0
        state = np.zeros((self.sequence_length, self.state_size))
        steps_done = 0
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
            steps_done += 1
            if done:
                print(f"⚠️ Fine-tuning stopped early at step {step} due to 'done' condition.")
                break

        print(f"✅ Fine-tuning complete for {steps_done} steps.")
        return steps_done
