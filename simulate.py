import cProfile
import argparse
import random
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', type=int, required=True)
parser.add_argument('--precompute_directory', type=str, required=True)
parser.add_argument('--gamedata_directory', type=str, required=True)
parser.add_argument('--num_playthroughs', type=int, default=10000)
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

MOVES = load_moves(args.precompute_directory)

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

        self.compute_extras = compute_extras
        if compute_extras:
            self.last_move_index = None

    def play_move(self, player, move_index):
        # Rule out some moves.
        self.moves_ruled_out |= MOVES["moves_ruled_out_for_all"][move_index]
        self.moves_ruled_out[player] |= MOVES["moves_ruled_out_for_player"][move_index]

        # Enable new moves for player based on corners.
        self.moves_enabled[player] |= MOVES["moves_enabled_for_player"][move_index]

        # Compute scores.
        self.accumulated_scores[player] += MOVES["scores"][move_index]

        # Update occupancies.
        self.occupancies[player] |= MOVES["new_occupieds"][move_index]

        if self.compute_extras:
            self.last_move_index = move_index

    def select_random_valid_move_index(self, player):
        valid_moves = ~self.moves_ruled_out[player] & self.moves_enabled[player]
        valid_move_indices = np.flatnonzero(valid_moves)
        if len(valid_move_indices) == 0:
            return None
        else:
            return np.random.choice(valid_move_indices)
    
    def scores(self):
        return self.accumulated_scores
    
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


def run(playthrough_count):
    print("Starting run")

    boards = []
    turns = []
    results = []
    
    for _ in tqdm(range(playthrough_count)):
        current_player = random.randint(0, 3)
        state = State()

        skipped_players = set()
        num_turns = 0

        while True:
            boards.append(state.occupancies.copy())
            turns.append(np.eye(4)[current_player])
            num_turns += 1

            move_index = state.select_random_valid_move_index(current_player)
            if move_index is None:
                skipped_players.add(current_player)
                if len(skipped_players) == 4:
                    break
            else:
                state.play_move(current_player, move_index)

            current_player = (current_player + 1) % 4

        result = [0, 0, 0, 0]
        result[np.argmax(state.scores())] = 1
        results.extend([result] * num_turns)

    boards = np.array(boards)
    turns = np.array(turns)
    results = np.array(results)

    samples = boards.shape[0]
    assert boards.shape == (samples, 4, BOARD_SIZE, BOARD_SIZE)
    assert turns.shape == (samples, 4)
    assert results.shape == (samples, 4)    

    print("Saving outputs to disk...")
    os.makedirs(args.gamedata_directory, exist_ok=True)
    
    # Save the data.
    np.save(f"{args.gamedata_directory}/boards.npy", boards)
    np.save(f"{args.gamedata_directory}/turns.npy", turns)
    np.save(f"{args.gamedata_directory}/results.npy", results)

run(args.num_playthroughs)