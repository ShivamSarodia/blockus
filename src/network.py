# The network accepts an occupancies array and outputs both 
# a policy prediction and value. It expects inputs and outputs
# to be in the perspective of the current player. (So e.g. the 
# occupancies array should be rolled and rotated to the current
# player's perspective.)

from torch import nn

from constants import BOARD_SIZE, NUM_MOVES

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # TODO: Do I want to softmax here? When training, I may want to have the raw values
        # depending on my loss function.
        self.generate_values = nn.Sequential(
            nn.Linear(256, 4),
            nn.Softmax(dim=1),
        )
        self.generate_children_search_distribution = nn.Sequential(
            nn.Linear(256, NUM_MOVES),
            nn.Softmax(dim=1),
        )

    def forward(self, occupancies):
        shared = self.shared_sequence(occupancies)
        return self.generate_values(shared), self.generate_children_search_distribution(shared)