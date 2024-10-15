# The network accepts an occupancies array and outputs both 
# a policy prediction and value. It expects inputs and outputs
# to be in the perspective of the current player. (So e.g. the 
# occupancies array should be rolled and rotated to the current
# player's perspective.)

import numpy as np
from typing import Tuple

import torch
from torch import nn

from constants import BOARD_SIZE, NUM_MOVES

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.generate_values = nn.Sequential(
            nn.Linear(256, 4),
        )
        self.generate_children_search_distribution = nn.Sequential(
            nn.Linear(256, NUM_MOVES),
        )

    def forward(self, occupancies):
        shared = self.shared_sequence(occupancies)
        return self.generate_values(shared), self.generate_children_search_distribution(shared)
    

def evaluate(model, occupancies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    occupancies_tensor = torch.from_numpy(occupancies).to(torch.float).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        raw_values, raw_policies = model(occupancies_tensor)
    return (
        torch.softmax(raw_values, dim=1).squeeze(0).numpy(),
        torch.softmax(raw_policies, dim=1).squeeze(0).numpy(),
    )