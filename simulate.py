import cProfile
import argparse
import random
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', type=int, required=True)
parser.add_argument('--precompute_dir', type=str, required=True)
# parser.add_argument('--games_dir', type=str, required=True)
# parser.add_argument('--num_playthroughs', type=int, default=10000)
args = parser.parse_args()

BOARD_SIZE = args.board_size

def load_moves(dir):
    # For every file in precomputed_moves directory ending in .npy,
    # read the file into dictionary.
    moves = {}
    filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    moves = {}
    for filename in filenames:
        if filename.endswith(".npy"):
            key = filename[:-4]
            with open(f"{dir}/{filename}", "rb") as f:
                print("Loading file:", key)
                moves[key] = np.load(f)

    return moves

MOVES = load_moves(args.precompute_dir)

NUM_MOVES = MOVES["new_occupieds"].shape[0]

print(f"Loaded {NUM_MOVES} moves.")

# Precompute the initial value of the self.moves_enabled array for a state.
start_corners = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
start_corners[0, 0, 0] = True
start_corners[1, 0, BOARD_SIZE - 1] = True
start_corners[2, BOARD_SIZE - 1, BOARD_SIZE - 1] = True
start_corners[3, BOARD_SIZE - 1, 0] = True
INITIAL_MOVES_ENABLED = np.any(MOVES["new_occupieds"] & start_corners[:, np.newaxis, :, :], axis=(2, 3))
assert INITIAL_MOVES_ENABLED.shape == (4, NUM_MOVES)

class State:
    def __init__(self, compute_extras=False):
        # Moves for each player that are permissible because they intersect
        # with an exposed corner.
        self.moves_enabled = INITIAL_MOVES_ENABLED.copy()        

        # Moves for each player that are ruled out by conflict
        # with an occupied square, conflict with a player's own adjacent
        # square, or having used the piece already.
        self.moves_ruled_out = np.zeros((4, NUM_MOVES), dtype=bool)

        # Compute the score as we go.
        self.accumulated_scores = np.zeros(4, dtype=int)

        # Track occupancies as the board state.
        self.occupancies = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)

        # Track which players are out of moves.
        self.players_out_of_moves = 0

        # Player 0 starts.
        self.player = 0

        self.compute_extras = compute_extras
        if compute_extras:
            self.last_move_index = None

    def play_move(self, move_index):
        # Assumes that the move_index provided is valid.
        # If move_index is None, skips the player's turn (only valid if no moves are available).
        if move_index is None:
            self.players_out_of_moves |= 1 << self.player
        else:
            # Update occupancies.
            self.occupancies[self.player] |= MOVES["new_occupieds"][move_index]            

            # Rule out some moves.
            self.moves_ruled_out |= MOVES["moves_ruled_out_for_all"][move_index]
            self.moves_ruled_out[self.player] |= MOVES["moves_ruled_out_for_player"][move_index]

            # Enable new moves for player based on corners.
            self.moves_enabled[self.player] |= MOVES["moves_enabled_for_player"][move_index]

            # Compute scores.
            self.accumulated_scores[self.player] += MOVES["scores"][move_index]

        if self.compute_extras:
            self.last_move_index = move_index        
        
        # Update player.
        self.player = (self.player + 1) % 4

    def select_random_valid_move_index(self):
        valid_moves = ~self.moves_ruled_out[self.player] & self.moves_enabled[self.player]
        valid_move_indices = np.flatnonzero(valid_moves)
        if len(valid_move_indices) == 0:
            return None
        else:
            return np.random.choice(valid_move_indices)
    
    def scores(self):
        return self.accumulated_scores
    
    def final_scores(self):
        # 15 = 1 + 2 + 4 + 8, i.e. 1111 in binary.
        if self.players_out_of_moves == 15:
            return self.scores()
    
    def pretty_print_board(self):
        assert self.compute_extras, "Must have compute_extras set to print board."

        grid = np.zeros((BOARD_SIZE, BOARD_SIZE, 3))
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.occupancies[0, x, y]:
                    color = [1, 0, 0]
                elif self.occupancies[1, x, y]:
                    color = [0, 1, 0]
                elif self.occupancies[2, x, y]:
                    color = [0.25, 0.25, 1]
                elif self.occupancies[3, x, y]:
                    color = [0, 1, 1]
                else:
                    color = [1, 1, 1]
                grid[y, x] = color

        # Plot the grid
        plt.imshow(grid, interpolation='nearest')
        plt.axis('on')  # Show the axes
        plt.grid(color='black', linestyle='-', linewidth=2)  # Add gridlines

        if self.last_move_index is not None:
            last_move_new_occupieds = MOVES["new_occupieds"][self.last_move_index]
            x_coords, y_coords = [], []
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if last_move_new_occupieds[x, y]:
                        x_coords.append(x)
                        y_coords.append(y)

            plt.scatter(x_coords, y_coords, color='black', s=20)  # Draw black dots

        # Adjust the gridlines to match the cells
        plt.xticks(np.arange(-0.5, BOARD_SIZE, 1), [])
        plt.yticks(np.arange(-0.5, BOARD_SIZE, 1), [])
        plt.gca().set_xticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.show()


def play_game():
    print("Starting game...")

    state = State()
    while True:
        move_index = state.select_random_valid_move_index()
        state.play_move(move_index)

        score = state.final_scores()
        if score is not None:
            return score

print(play_game())