import numpy as np
import json

from state import State
from configuration import config, moves_data

BOARD_SIZE = config()["game"]["board_size"]
MOVES_DATA = moves_data()

class RandomAgent:
    async def select_move_index(self, state: State):
        # Get all valid moves.
        valid_moves = np.flatnonzero(state.valid_moves_array())
        
        # Find the move with the highest number of newly occupied squares.
        best_move_score = -1
        for move_index in valid_moves:
            best_move_score = max(best_move_score, np.sum(MOVES_DATA["new_occupieds"][move_index]))

        # Select a move randomly from all the best moves.
        best_moves = []
        for move_index in valid_moves:
            if np.sum(MOVES_DATA["new_occupieds"][move_index]) == best_move_score:
                best_moves.append(move_index)

        return np.random.choice(best_moves)


class HumanAgent:
    async def select_move_index(self, state: State):
        # Convert the state into the json representation for the UI.

        moves_ruled_out = state.moves_ruled_out[state.player]
        moves_not_ruled_out = np.flatnonzero(~moves_ruled_out)
        pieces_available = np.unique(MOVES_DATA["piece_indices"][moves_not_ruled_out])

        # TODO: DRY this, maybe store it in MOVES_DATA.
        PIECES = [
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
            [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
            [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
            [(1, 1), (0, 1), (1, 0), (2, 1), (1, 2)],
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
            [(0, 0), (1, 0), (1, 1), (2, 1)],
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (1, 0), (2, 0), (0, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 0)],
            [(0, 0), (1, 0), (2, 0), (1, 1)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (1, 0), (0, 1)],
            [(0, 0), (1, 0)],
            [(0, 0)],
        ]

        pieces = []
        for piece_index in pieces_available:
            pieces.append(PIECES[piece_index])

        print("Encoded state:")
        print(json.dumps(
            {
                "board": state.occupancies.astype(int).tolist(),
                "player": state.player,
                "pieces": pieces,
            }
        ))

        move_encoded = input("Enter encoded move: ")

        move_coordinates_occupied = json.loads(move_encoded)
        move_new_occupieds = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        for x, y in move_coordinates_occupied:
            move_new_occupieds[x, y] = True

        move_index = np.where(np.all(MOVES_DATA["new_occupieds"] == move_new_occupieds, axis=(1, 2)))[0][0]
        return move_index
