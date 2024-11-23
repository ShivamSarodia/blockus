import numpy as np

from state import State
from configuration import moves_data

MOVES_DATA = moves_data()

class CustomAgent:
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
