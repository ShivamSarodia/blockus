import random
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Pool

from moves import Move
from generate_moves import ALL_PIECES
from load_moves import load_moves

BOARD_SIZE = 20
PIECE_MOVES = load_moves()

class State:
    def __init__(self, occupancies=None, adjacent=None, corners=None, remaining_pieces=None):
        self.occupancies = occupancies or [0] * 4
        self.adjacent = adjacent or [0] * 4

        self.corners = corners or [0] * 4
        self.corners[0] |= (1 << ((0 + 0 * BOARD_SIZE)))
        self.corners[1] |= (1 << ((0 + 19 * BOARD_SIZE)))
        self.corners[2] |= (1 << ((19 + 19 * BOARD_SIZE)))
        self.corners[3] |= (1 << ((19 + 0 * BOARD_SIZE)))

        self.remaining_pieces = remaining_pieces or {
            0: set(range(len(ALL_PIECES))),
            1: set(range(len(ALL_PIECES))),
            2: set(range(len(ALL_PIECES))),
            3: set(range(len(ALL_PIECES))),
        }

        self.last_move = None

    def copy(self):
        return State(
            occupancies=[o for o in self.occupancies],
            adjacent=[a for a in self.adjacent],
            corners=[c for c in self.corners],
            remaining_pieces={k: set(v) for k, v in self.remaining_pieces.items()}
        )

    def is_valid_move(self, player, move):
        if move.piece_index not in self.remaining_pieces[player]:
            return False

        for occupancy in self.occupancies:
            if occupancy & move.new_occupied:
                return False
        
        if self.corners[player] & move.new_occupied == 0:
            return False
        
        if self.adjacent[player] & move.new_occupied:
            return False
        
        return True
    
    def compute_actually_available_corners(self, player):
        available_corners_bitstring = self.corners[player] & (~self.adjacent[player])
        for player in range(4):
           available_corners_bitstring &= (~self.occupancies[player])
        return available_corners_bitstring
    
    def compute_empty_adjacents(self, player):
        empty_adjacents_bitstring = self.adjacent[player]
        for player in range(4):
           empty_adjacents_bitstring &= (~self.occupancies[player])
        return empty_adjacents_bitstring
    
    def get_valid_moves(self, player):
        valid_moves = []
        for piece_index in self.remaining_pieces[player]:
            for move in PIECE_MOVES[piece_index]:
                if self.is_valid_move(player, move):
                    valid_moves.append(move)
        return valid_moves
    
    def play_move(self, player, move):
        self.occupancies[player] |= move.new_occupied
        self.corners[player] |= move.new_corners
        self.adjacent[player] |= move.new_adjacents
        self.remaining_pieces[player].remove(move.piece_index)

        self.last_move = move

    def scores(self):
        scores = [0, 0, 0, 0]
        for player in range(4):
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if self.occupancies[player] & (1 << (x + y * BOARD_SIZE)):
                        scores[player] += 1
        return scores

    def pretty_print_board(self):
        grid_size = 20
        grid = np.zeros((grid_size, grid_size, 3))
        for x in range(grid_size):
            for y in range(grid_size):
                if self.occupancies[0] & (1 << (x + y * BOARD_SIZE)):
                    color = [1, 0, 0]
                elif self.occupancies[1] & (1 << (x + y * BOARD_SIZE)):
                    color = [0, 1, 0]
                elif self.occupancies[2] & (1 << (x + y * BOARD_SIZE)):
                    color = [0.25, 0.25, 1]
                elif self.occupancies[3] & (1 << (x + y * BOARD_SIZE)):
                    color = [0, 1, 1]
                else:
                    color = [1, 1, 1]
                grid[x,  y] = color

        # Plot the grid
        plt.imshow(grid, interpolation='nearest')
        plt.axis('on')  # Show the axes
        plt.grid(color='black', linestyle='-', linewidth=2)  # Add gridlines

        dot_positions = self.last_move.display_coordinates

        # Extract the x and y coordinates for scatter plot
        x_coords = [col for row, col in dot_positions]
        y_coords = [row for row, col in dot_positions]

        plt.scatter(x_coords, y_coords, color='black', s=20)  # Draw black dots

        # Adjust the gridlines to match the cells
        plt.xticks(np.arange(-0.5, grid_size, 1), [])
        plt.yticks(np.arange(-0.5, grid_size, 1), [])
        plt.gca().set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.show()


def pick_random_move(moves, player, state):
    return random.choice(moves)


def score_board(state, player, weights):
    num_occupied = bin(state.occupancies[player]).count('1')
    num_available_corners = bin(state.compute_actually_available_corners(player)).count('1')
    num_available_adjacents = bin(state.compute_empty_adjacents(player)).count('1')

    return weights["num_occupied"] * num_occupied + weights["num_available_corners"] * num_available_corners - weights["num_available_adjacents"] * num_available_adjacents


def pick_scored_move(moves, player, state, weights):
    moves_and_scores = []

    for move in moves:
        new_state = state.copy()
        new_state.play_move(player, move)

        new_scores = {player: score_board(new_state, player, weights) for player in range(4)}

        average_other_player_score = (sum(new_scores.values()) - new_scores[player]) / (len(new_scores) - 1)
        relative_score = new_scores[player] - weights["average_other_player_score"] * average_other_player_score

        moves_and_scores.append((move, relative_score))

    moves_and_scores = sorted(moves_and_scores, key=lambda x: x[1], reverse=True)
    return random.choice(moves_and_scores[:4])[0]


weights_per_player = [
    {
        "num_occupied": 1,
        "num_available_corners": 0.25,
        "num_available_adjacents": 0,
        "average_other_player_score": 1,
    },
    {
        "num_occupied": 1,
        "num_available_corners": 0.125,
        "num_available_adjacents": 0,
        "average_other_player_score": 1,
    },
    {
        "num_occupied": 1,
        "num_available_corners": 0.5,
        "num_available_adjacents": 0,
        "average_other_player_score": 1,
    },
    {
        "num_occupied": 1,
        "num_available_corners": 0.25,
        "num_available_adjacents": 0,
        "average_other_player_score": 1,
    },
]


def run(count):
    print("starting run")
    wins = [0, 0, 0, 0]
    
    for _ in range(count):
        current_player = random.randint(0, 3)
        state = State()

        skipped_players = set()

        while True:
            moves = state.get_valid_moves(current_player)
            if len(moves) == 0:
                skipped_players.add(current_player)
                if len(skipped_players) == 4:
                    break
            else:
                # num_moves.append(len(moves))
                weights = weights_per_player[current_player]
                move = pick_scored_move(moves, current_player, state, weights)
                state.play_move(current_player, move)
            
            current_player = (current_player + 1) % 4

        scores = state.scores()
        wins[scores.index(max(scores))] += 1
    
    return wins

def main():
    with Pool(processes=4) as pool:
        results = pool.map(run, [25] * 4)

    final_result = [0, 0, 0, 0]
    for result in results:
        for i, count in enumerate(result):
            final_result[i] += count

    print(final_result)

if __name__ == "__main__":
    main()
