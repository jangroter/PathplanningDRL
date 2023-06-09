from math import gamma
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.modules import Module

import torch.nn as nn
from torch.distributions import Normal

NETWORK_STRUCTURE = [[256,torch.nn.ReLU()],[256,torch.nn.ReLU()],[256,torch.nn.ReLU()]]
# NETWORK_STRUCTURE = [[2048,torch.nn.ReLU()],[2048,torch.nn.ReLU()]]

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
        log_std_min: float= -20,
        log_std_max: float=2):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layers = nn.ModuleList()
        self.in_dim = in_dim

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

        return mu, log_prob

class Agent:
    def __init__(self, action_dim = 2, state_dim = 2, modelname = "actor.pt"):
        self.statedim = state_dim
        self.actiondim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.statedim, self.actiondim, NETWORK_STRUCTURE).to(self.device)

        self.actor.load_state_dict(torch.load(modelname, map_location=torch.device('cpu')))

    def step(self, state):
        selected_action = []
        action = self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
        selected_action.append(action)
        selected_action = np.array(selected_action)
        selected_action = np.clip(selected_action, -1, 1)
        return selected_action

class Value:
    def __init__(self, action_dim = 2, state_dim = 2, modelname = "vf.pt"):
        self.statedim = state_dim
        self.actiondim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vf = CriticV(self.statedim, NETWORK_STRUCTURE).to(self.device)
        self.vf.load_state_dict(torch.load(modelname, map_location=torch.device('cpu')))

    def step(self, state):
        value = self.vf.forward(torch.FloatTensor(state).to(self.device))
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