import torch
import numpy as np

from model import LinearDeepQModel


class Agent:
    def __init__(self, input_dims, n_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        # Pass parameters
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        # Create action space and Q
        self.action_space = [i for i in range(self.n_actions)]
        self.Q = LinearDeepQModel(input_dims=self.input_dims, n_actions=self.n_actions, layer_dims=[128, 128],
                                  learning_rate=self.learning_rate)

    # Create function for to choose action
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).to(self.Q.device)
            action = torch.argmax(self.Q.forward(state)).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        # Gradients reset
        self.Q.optimizer.zero_grad()
        # Move to GPU or CPU
        states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(state_, dtype=torch.float).to(self.Q.device)

        Q_action_prediction = self.Q.forward(states)[actions]
        Q_next_state = self.Q.forward(states_).max()
        Q_target = reward + self.gamma * Q_next_state

        loss = self.Q.loss(Q_target, Q_action_prediction).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
