import cProfile
import numpy as np
import random
from load_moves import load_moves
from matplotlib import pyplot as plt
from tqdm import tqdm

BOARD_SIZE = 20

MOVES = load_moves()
NUM_MOVES = MOVES["new_occupieds"].shape[0]

print(f"Loaded {NUM_MOVES} moves.")

# Precompute the initial value of the self.moves_enabled array for a state.
start_corners = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
start_corners[0, 0, 0] = True
start_corners[1, 0, 19] = True
start_corners[2, 19, 19] = True
start_corners[3, 19, 0] = True
INITIAL_MOVES_ENABLED = np.any(MOVES["new_occupieds"] & start_corners[:, np.newaxis, :, :], axis=(2, 3))
assert INITIAL_MOVES_ENABLED.shape == (4, NUM_MOVES)

class State:
    def __init__(self, compute_extras=False):
        # For each player, one corner is initially playable.
        start_corners = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        start_corners[0, 0, 0] = True
        start_corners[1, 0, 19] = True
        start_corners[2, 19, 19] = True
        start_corners[3, 19, 0] = True

        # Moves for each player that are permissible because they intersect
        # with an exposed corner.
        self.moves_enabled = INITIAL_MOVES_ENABLED.copy()        

        # Moves for each player that are ruled out by conflict
        # with an occupied square, conflict with a player's own adjacent
        # square, or having used the piece already.
        self.moves_ruled_out = np.zeros((4, NUM_MOVES), dtype=bool)

        # Compute the score as we go.
        self.accumulated_scores = np.zeros(4, dtype=int)

        self.compute_extras = compute_extras
        if compute_extras:
            self.occupancies = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
            self.last_move_index = None

    def play_move(self, player, move_index):
        # Rule out some moves.
        self.moves_ruled_out |= MOVES["moves_ruled_out_for_all"][move_index]
        self.moves_ruled_out[player] |= MOVES["moves_ruled_out_for_player"][move_index]

        # Enable new moves for player based on corners.
        self.moves_enabled[player] |= MOVES["moves_enabled_for_player"][move_index]

        # Compute scores.
        self.accumulated_scores[player] += MOVES["scores"][move_index]

        if self.compute_extras:
            self.occupancies[player] |= MOVES["new_occupieds"][move_index]
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


def run(count):
    print("starting run")
    wins = np.zeros(4)
    
    for _ in tqdm(range(count)):
        current_player = random.randint(0, 3)
        state = State()

        skipped_players = set()

        while True:
            move_index = state.select_random_valid_move_index(current_player)
            if move_index is None:
                skipped_players.add(current_player)
                if len(skipped_players) == 4:
                    break
            else:
                state.play_move(current_player, move_index)
            
            current_player = (current_player + 1) % 4

            # state.pretty_print_board()

        wins[np.argmax(state.scores())] += 1
    
    return wins

cProfile.run("run(1000)")