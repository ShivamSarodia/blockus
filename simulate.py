import cProfile
import argparse
import time
import datetime
import random
import os
from typing import Dict, NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', type=int, required=True)
parser.add_argument('--precompute_dir', type=str, required=True)
parser.add_argument('--debug_mode', action='store_true')
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
    def __init__(self):
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

        # Index of the previous move played, useful for rendering the game.
        if args.debug_mode:
            self.last_move_index = None

    def clone(self):
        new_state = State()
        new_state.moves_enabled = self.moves_enabled.copy()
        new_state.moves_ruled_out = self.moves_ruled_out.copy()
        new_state.accumulated_scores = self.accumulated_scores.copy()
        new_state.occupancies = self.occupancies.copy()
        new_state.player = self.player

        if args.debug_mode:      
            new_state.last_move_index = self.last_move_index

        return new_state

    def play_move(self, move_index) -> bool:
        """
        Play the given move and update self.player to the new player (skipping any players 
        who are out of moves). Return whether the game is over.

        This method assumes the provided move is valid.
        """
        if args.debug_mode and not self.valid_moves_array()[move_index]:
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

        if args.debug_mode:
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
        assert args.debug_mode, "Must have debug_mode set to print board."
        Display(self.occupancies, MOVES["new_occupieds"][self.last_move_index]).show()
        
    
class Display:
    def __init__(self, occupancies, overlay_dots=None):
        self.occupancies = occupancies
        self.overlay_dots = overlay_dots

    def show(self):
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

        if self.overlay_dots is not None:
            x_coords, y_coords = [], []
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if self.overlay_dots[x, y]:
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


def scores_to_values(scores):
    # The value is 1 for all winning players, and -1 for all losing players.
    return np.where(scores == np.max(scores), 1, -1)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# TODO: Move these into a normalizers.py or something.

# A negative value in these methods moves from player POV 
# back to universal POV.
def occupancies_to_player_pov(occupancies, player):
    rotated_board = np.rot90(occupancies, k=player, axes=(-2, -1))
    return np.roll(rotated_board, shift=-player, axis=-3) 


def moves_array_to_player_pov(moves_array, player):
    return moves_array[MOVES["rotation_mapping"][(-player) % 4]]


def values_to_player_pov(values, player):
    np.roll(values, shift=-player, axis=-1)


class Config(NamedTuple):
    num_mcts_rollouts: int
    ucb_pb_c_base: float 
    ucb_pb_c_init: float 
    root_dirichlet_alpha: float
    root_exploration_fraction: float


class MCTSAgent:
    def __init__(self, config: Config):
        self.config = config

    def select_move_index(self, state: State):
        search_root = MCTSValuesNode(self.config)
        search_root.get_value_and_expand_children(state)

        for _ in range(self.config.num_mcts_rollouts):
            scratch_state = state.clone()

            # At the start of each iteration, nodes_visited will be one 
            # longer than moves_played. moves_played[i] is the move played
            # to exit nodes_visited[i].
            nodes_visited = [search_root]
            moves_played = []

            while True:
                move_index = nodes_visited[-1].select_child_by_ucb(scratch_state)
                moves_played.append(move_index)

                game_over = scratch_state.play_move(move_index)
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
                value = scratch_state.scores()
            else:
                new_node = MCTSValuesNode(self.config)
                value = new_node.get_value_and_expand_children(scratch_state)
                nodes_visited[-1].move_index_to_child_node[moves_played[-1]] = new_node

            # Now, backpropagate the value up the visited notes.
            for i in range(len(nodes_visited)):
                nodes_visited[i].children_value_sums[:,moves_played[i]] += value 
                nodes_visited[i].children_visit_counts[moves_played[i]] += 1

        # Save the search tree results to the data buffer.
        # TODO: We need to figure out the right place to define the interface here. Because the
        # get_value_and_expand_children operation needs to revert whatever operation we perform
        # on the state to get into the NN image/target state.
        # data_buffer.report_node_search_distribution(state, search_root.children_visit_counts)

        # Select the move to play now.
        return search_root.get_move_index_to_play(state)
    

class DataBuffer:
    def __init__(self):
        self.occupancies = []
        self.children_visit_distributions = []
        self.values = []

    def report_turn(self, player_pov_occupancies: np.ndarray, player_pov_children_visit_distribution: np.ndarray):
        self.occupancies.append(player_pov_occupancies)
        self.children_visit_distributions.append(player_pov_children_visit_distribution)
    
    def report_game_end(self, values: np.ndarray):
        for _ in range(len(self.occupancies) - len(self.values)):
            self.values.append(values)

        self._maybe_flush_to_disk()

    def _maybe_flush_to_disk(self):
        if len(self.occupancies) < 1000:
            return
        
        key = str(time.time())        
        np.save("occupancies_.npy", np.array(self.occupancies))

        # Write the data to disk.


class MCTSValuesNode:
    def __init__(self, config: Config):
        self.config = config

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
        pb_c = np.log((total_visit_count + self.config.ucb_pb_c_base + 1) / self.config.ucb_pb_c_base) + self.config.ucb_pb_c_init

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
        ucb_scores[~state.valid_moves_array()] = -np.inf

        return np.argmax(ucb_scores)

    def get_value_and_expand_children(self, state: State):
        """
        Populate the children arrays by calling the NN, and return
        the value of the current state.
        """
        # TODO: This is supposed to be input to the NN.
        player_pov_occupancies = state.player_pov_occupancies()

        # TODO: These are supposed to be output from the NN.
        player_pov_values = 2 * np.random.random_sample((4,)) - 1
        player_pov_children_priors = np.random.random_sample((NUM_MOVES,))

        # Rotate the player POV values and policy back to the original player's perspective.
        universal_values = np.roll(player_pov_values, shift=state.player, axis=0)
        universal_children_priors = player_pov_children_priors[MOVES["rotation_mapping"][state.player]]

        self.children_value_sums = np.zeros((4, NUM_MOVES), dtype=float)
        self.children_visit_counts = np.zeros(NUM_MOVES, dtype=int)
        self.children_priors = universal_children_priors
        return universal_values

    def get_move_index_to_play(self, state: State):
        probabilities = softmax(self.children_visit_counts)
        probabilities *= state.valid_moves_array()
        probabilities /= np.sum(probabilities)

        return np.random.choice(len(probabilities), p=probabilities)
    
    def add_noise(self):
        self.children_priors = (1 - self.config.root_exploration_fraction) * self.children_priors + \
            self.config.root_exploration_fraction * np.random.gamma(self.config.root_dirichlet_alpha, 1, NUM_MOVES)


class RandomAgent:
    def select_move_index(self, state: State):
        return state.select_random_valid_move_index()  


def play_game():
    print("Playing game...")

    config = Config(
        num_mcts_rollouts=500,
        ucb_pb_c_base=1.25,
        ucb_pb_c_init=1.25,
        root_dirichlet_alpha=0.03,
        root_exploration_fraction=0.25
    )

    agents = [
        # MCTSAgent(config),
        # MCTSAgent(config),
        # MCTSAgent(config),
        # MCTSAgent(config),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    game_over = False
    state = State()
    while not game_over:
        agent = agents[state.player]
        move_index = agent.select_move_index(state)
        game_over = state.play_move(move_index)

        board_zero_position = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        board_zero_position[0, 0] = True
        Display(state.player_pov_occupancies(), board_zero_position).show()

    # values = scores_to_values(state.scores())
    # return values[mcts_agent_index], sum(values) / 4

# start_time = datetime.datetime.now()
# [play_game() for _ in range(5)]
# print(datetime.datetime.now() - start_time)

# mcts_scores = []
# avg_scores = []
# for _ in range(50):
#     mcts_score, avg_score = play_game()
#     mcts_scores.append(mcts_score)
#     avg_scores.append(avg_score)

# print("Average MCT score:", sum(mcts_scores) / len(mcts_scores))
# print("Average score:", sum(avg_scores) / len(avg_scores))

# play_game()

# Write some code here to test rotation mappings.

# state = State()
# for _ in range(13):
#     state.play_move(state.select_random_valid_move_index())

# player_pov_move_index = 20
# player_pov_moves = np.zeros((NUM_MOVES,))
# player_pov_moves[player_pov_move_index] = 1

# universal_moves = player_pov_moves[MOVES["rotation_mapping"][state.player]]
# universal_move_index = np.argmax(universal_moves)

# # The board when dots for player_pov_move_index occupancy is added onto player_pov_occupancies
# # should match the board when dots for universal_move_index occupancy is added onto occupancies.

# Display(
#     state.occupancies,
#     MOVES["new_occupieds"][universal_move_index]
# ).show()

# Display(
#     state.player_pov_occupancies(),
#     MOVES["new_occupieds"][player_pov_move_index]
# ).show()