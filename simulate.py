import cProfile
import argparse
import random
import os
from typing import Dict
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

        # Player 0 starts.
        self.player = 0

        # TODO: If you add more here, add it to clone too.

        self.compute_extras = compute_extras
        if compute_extras:
            self.last_move_index = None

    def clone(self):
        new_state = State()
        new_state.moves_enabled = self.moves_enabled.copy()
        new_state.moves_ruled_out = self.moves_ruled_out.copy()
        new_state.accumulated_scores = self.accumulated_scores.copy()
        new_state.occupancies = self.occupancies.copy()
        new_state.player = self.player

        if self.compute_extras:
            new_state.compute_extras = self.compute_extras            
            new_state.last_move_index = self.last_move_index

        return new_state

    def play_move(self, move_index) -> bool:
        """
        Play the given move and update self.player to the new player (skipping any players 
        who are out of moves). Return whether the game is over.

        This method assumes the provided move is valid.
        """
        if self.compute_extras and not self.valid_moves_array()[move_index]:
            raise "Playing an invalid move!"

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
        
        # Find the next player who has a valid move.
        for _ in range(4):
            self.player = (self.player + 1) % 4
            if self.valid_moves_array().any():
                return False
            
        return True

    def valid_moves_array(self):
        return ~self.moves_ruled_out[self.player] & self.moves_enabled[self.player]

    def select_random_valid_move_index(self):
        valid_move_indices = np.flatnonzero(self.valid_moves_array())
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


NUM_MCTS_ROLLOUTS = 100
UCB_PB_C_BASE = 10
UCB_PB_C_INIT = 1.25

class MCTSValuesNode:
    def __init__(self):
        # All of these below can be rewritten in terms of num *valid* moves to save a lot
        # of memory if needed.
        #
        # They are populated when get_value_and_expand_children is called.
        self.children_value_sums = None      # Shape (4, NUM_MOVES)
        self.children_visit_counts = None    # Shape (NUM_MOVES,)
        self.children_priors = None          # Shape (NUM_MOVES,)

        # This populates over time only as nodes are visited.
        self.move_index_to_child_node: Dict[int, MCTSValuesNode] = {}

    def select_child_by_ucb(self, state: State):
        total_visit_count = np.sum(self.children_visit_counts)

        # Just a number so far
        pb_c = np.log((total_visit_count + UCB_PB_C_BASE + 1) / UCB_PB_C_BASE) + UCB_PB_C_INIT

        # Now, it's an array for each child.
        pb_c *= np.sqrt(total_visit_count) / (self.children_visit_counts + 1)

        prior_scores = pb_c * self.children_priors 
        value_scores = np.divide(
            self.children_value_sums[state.player],
            self.children_visit_counts,
            out=np.zeros_like(self.children_value_sums[state.player]),
            where=(self.children_visit_counts != 0)
        )

        ucb_scores = prior_scores + value_scores

        import ipdb; ipdb.set_trace()

        ucb_scores[~state.valid_moves_array()] = -np.inf

        return np.argmax(ucb_scores)

    def get_value_and_expand_children(self, state: State):
        # Populate the children arrays. 
        # TODO: Insert a neural network here instead!
        self.children_value_sums = np.zeros((4, NUM_MOVES), dtype=float)
        self.children_visit_counts = np.zeros(NUM_MOVES, dtype=int)
        self.children_priors = np.random.random_sample((NUM_MOVES,))

        # TODO: We should let the NN evaluate the board here. Instead,
        # I'm estimating the scores as the value of this node.
        return state.scores()

    def get_move_index_to_play(self, state: State):
        # TODO: For early moves, we should sample with softmax instead.
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.divide(
                self.children_value_sums[state.player],
                self.children_visit_counts,
            )
        return np.nanargmax(vals)

def select_move_index(original_state):
    # print(f"Select move index called for state with player {original_state.player}")

    search_root = MCTSValuesNode()
    search_root.get_value_and_expand_children(original_state)
    # TODO: Add noise

    for _ in range(NUM_MCTS_ROLLOUTS):
        state = original_state.clone()

        # At the start of each iteration, nodes_visited will be one 
        # longer than moves_played. moves_played[i] is the move played
        # to exit nodes_visited[i].
        nodes_visited = [search_root]
        moves_played = []

        while True:
            move_index = nodes_visited[-1].select_child_by_ucb(state)
            moves_played.append(move_index)

            game_over = state.play_move(move_index)
            # print("Move played:", move_index)
            # state.pretty_print_board()

            # If the game is over, we can now assign values based on the final state.
            if game_over:
                break

            next_node = nodes_visited[-1].move_index_to_child_node.get(move_index)

            # If next_node does not exist, we need to break out of this loop to create that node and
            # fetch a new value to backpropagate.
            if not next_node:
                break

            nodes_visited.append(next_node)

        if game_over:
            value = state.scores()
        else:
            new_node = MCTSValuesNode()
            value = new_node.get_value_and_expand_children(state)
            nodes_visited[-1].move_index_to_child_node[moves_played[-1]] = new_node

        # Now, backpropagate the value up the visited notes.
        for i in range(len(nodes_visited)):
            nodes_visited[i].children_value_sums[:,moves_played[i]] += value 
            nodes_visited[i].children_visit_counts[moves_played[i]] += 1

    # Select the move to play now.
    return search_root.get_move_index_to_play(original_state)


# def select_move_index(state: State):
#     return state.select_random_valid_move_index()


def play_game():
    print("Starting game...")

    game_over = False

    state = State(compute_extras=False)
    while not game_over:
        move_index = select_move_index(state)
        game_over = state.play_move(move_index)
    
    return state.scores()

print(play_game())