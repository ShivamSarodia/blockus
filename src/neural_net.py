# The network accepts an occupancies array and outputs both 
# a policy prediction and value. It expects inputs and outputs
# to be in the perspective of the current player. (So e.g. the 
# occupancies array should be rolled and rotated to the current
# player's perspective.)

import numpy as np
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from constants import BOARD_SIZE, NUM_MOVES

# This is the original network.
# MAIN_BODY_CHANNELS = 256
# VALUE_HEAD_CHANNELS = 8
# VALUE_HEAD_FLAT_LAYER_WIDTH = 256
# POLICY_HEAD_CHANNELS = 256
# RESIDUAL_BLOCKS = 19

# Define some network architecture parameters
MAIN_BODY_CHANNELS = 128
VALUE_HEAD_CHANNELS = 32
VALUE_HEAD_FLAT_LAYER_WIDTH = 128
POLICY_HEAD_CHANNELS = 128
RESIDUAL_BLOCKS = 19

class Debug(nn.Module):
    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def forward(self, x):
        print(self.label, x.shape)
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels=MAIN_BODY_CHANNELS, out_channels=MAIN_BODY_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(MAIN_BODY_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(in_channels=MAIN_BODY_CHANNELS, out_channels=MAIN_BODY_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(MAIN_BODY_CHANNELS),
        )
    
    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=MAIN_BODY_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(MAIN_BODY_CHANNELS),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock()
            for _ in range(RESIDUAL_BLOCKS)
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=MAIN_BODY_CHANNELS, out_channels=VALUE_HEAD_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(VALUE_HEAD_CHANNELS),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(VALUE_HEAD_CHANNELS * BOARD_SIZE * BOARD_SIZE, VALUE_HEAD_FLAT_LAYER_WIDTH),
            nn.ReLU(),
            nn.Linear(VALUE_HEAD_FLAT_LAYER_WIDTH, 4),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=MAIN_BODY_CHANNELS, out_channels=POLICY_HEAD_CHANNELS, kernel_size=1),
            nn.BatchNorm2d(POLICY_HEAD_CHANNELS),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE * POLICY_HEAD_CHANNELS, NUM_MOVES),
        )

    def forward(self, occupancies):
        x = self.convolutional_block(occupancies)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return self.value_head(x), self.policy_head(x)
    

def evaluate(model, occupancies: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    occupancies_tensor = torch.from_numpy(occupancies).to(dtype=torch.float, device=device)
    model.eval()
    with torch.inference_mode():
        raw_values, raw_policies = model(occupancies_tensor)
    return (
        torch.softmax(raw_values, dim=1).cpu().numpy(),
        raw_policies.cpu().numpy(),
    )