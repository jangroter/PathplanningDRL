import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.distributions import Normal

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers_data: list,
        test: bool= False,
        log_std_min: float= -20,
        log_std_max: float=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layers = nn.ModuleList()
        self.in_dim = in_dim

        self.test = test

        # TODO maybe I should initialize the hidden_layers also with init_layer uniform
        for size, activation in layers_data:
                self.layers.append(nn.Linear(in_dim, size))
                in_dim = size
                if activation is not None:
                    assert isinstance(activation, Module), \
                        "Each tuple should contain a size (int) and a torch.nn.modules.Module"
                    self.layers.append(activation)
        print(layers_data[-1][0])
        log_std_layer = nn.Linear(layers_data[-1][0], out_dim)
        self.log_std_layer = init_layer_uniform(log_std_layer)

        mu_layer = nn.Linear(layers_data[-1][0], out_dim)
        self.mu_layer = init_layer_uniform(mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        input = state
        for layer in self.layers:
            input = layer(input)

        mu =  self.mu_layer(input).tanh()

        log_std = self.log_std_layer(input).tanh()
        log_std = self.log_std_min  + 0.5 * (
            self.log_std_max - self.log_std_min
            ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        if self.test:
            return mu, log_prob
        return action, log_prob


class CriticQ(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layers_data: list):
        super().__init__()

        self.layers = nn.ModuleList()
        self.in_dim = in_dim

        for size, activation in layers_data:
                self.layers.append(nn.Linear(in_dim, size))
                in_dim = size
                if activation is not None:
                    assert isinstance(activation, Module), \
                        "Each tuple should contain a size (int) and a torch.nn.modules.Module"
                    self.layers.append(activation)

        self.out = nn.Linear(layers_data[-1][0], 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state:torch.Tensor, 
        action: torch.Tensor) -> torch.Tensor:

        input = torch.cat((state, action), dim=-1)
        for layer in self.layers:
            input = layer(input)
        
        value = self.out(input)

        return value

class CriticV(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layers_data: list):
        super().__init__()

        self.layers = nn.ModuleList()
        self.in_dim = in_dim

        for size, activation in layers_data:
                self.layers.append(nn.Linear(in_dim, size))
                in_dim = size
                if activation is not None:
                    assert isinstance(activation, Module), \
                        "Each tuple should contain a size (int) and a torch.nn.modules.Module"
                    self.layers.append(activation)

        self.out = nn.Linear(layers_data[-1][0], 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, 
        state: torch.Tensor) -> torch.Tensor:

        input = state
        for layer in self.layers:
            input = layer(input)

        value = self.out(input)

        return value