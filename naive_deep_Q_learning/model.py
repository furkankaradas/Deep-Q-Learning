import torch

from torch import nn
from torch import optim
from torch.nn import functional as funct


class LinearDeepQModel(nn.Module):
    def __init__(self, n_actions, learning_rate, input_dims, layer_dims, cuda=True):
        super(LinearDeepQModel, self).__init__()
        # Pass the variables
        self.cuda = cuda
        self.n_actions = n_actions
        self.layer_dims = layer_dims
        self.input_dims = input_dims
        self.learning_rate = learning_rate
        # Create a model layers
        self.model_layers = self.create_model_layers()
        # Create a optimizer, loss
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        # Decide GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        self.to(self.device)

    def __str__(self):
        return "{}".format(self.model_layers)

    def create_model_layers(self):
        layers = nn.ModuleList([nn.Linear(*self.input_dims, self.layer_dims[0])])
        layers.extend([nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]) for i in range(len(self.layer_dims) - 1)])
        layers.append(nn.Linear(self.layer_dims[-1], self.n_actions))
        return layers

    def forward(self, x):
        for i in range(len(self.model_layers)):
            x = funct.relu(self.model_layers[i](x))
        return x
