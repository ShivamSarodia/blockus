import numpy as np
import functools

from configuration import config, moves_data
from display import Display


MOVES = moves_data()
NUM_MOVES = config()["game"]["num_moves"]
NUM_PIECES = config()["game"]["num_pieces"]
BOARD_SIZE = config()["game"]["board_size"]
DEBUG_MODE = config()["development"]["debug_mode"]


class State:
    def __init__(self):
        # Moves for each player that are permissible because they intersect
        # with an exposed corner.
        self.moves_enabled = self._get_initial_moves_enabled().copy()

        # Moves for each player that are ruled out by conflict
        # with an occupied square, conflict with a player's own adjacent
        # square, or having used the piece already.
        self.moves_ruled_out = np.zeros((4, NUM_MOVES), dtype=bool)

        # Compute the score as we go.
        self.accumulated_scores = np.zeros(4, dtype=int)

        # Track occupancies as the board state.
        self.occupancies = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)

        # Track unused pieces per player.
        self.unused_pieces = np.ones((4, NUM_PIECES), dtype=bool)
        
        # Player 0 starts.
        self.player = 0

        # Index of the previous move played, useful for rendering the game.
        if DEBUG_MODE:
            self.last_move_index = None

    @staticmethod
    @functools.cache
    def _get_initial_moves_enabled():
        print("Computing initial moves enabled...")

        # Precompute some initial values of the self.moves_enabled array for a state.
        start_corners = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        start_corners[0, 0, 0] = True
        start_corners[1, 0, BOARD_SIZE - 1] = True
        start_corners[2, BOARD_SIZE - 1, BOARD_SIZE - 1] = True
        start_corners[3, BOARD_SIZE - 1, 0] = True
        return np.any(MOVES["new_occupieds"] & start_corners[:, np.newaxis, :, :], axis=(2, 3))

    def clone(self):
        new_state = State()
        new_state.moves_enabled = self.moves_enabled.copy()
        new_state.moves_ruled_out = self.moves_ruled_out.copy()
        new_state.accumulated_scores = self.accumulated_scores.copy()
        new_state.occupancies = self.occupancies.copy()
        new_state.unused_pieces = self.unused_pieces.copy()
        new_state.player = self.player

        if DEBUG_MODE:
            new_state.last_move_index = self.last_move_index

        return new_state

    def play_move(self, move_index) -> bool:
        """
        Play the given move and update self.player to the new player (skipping any players 
        who are out of moves). Return whether the game is over.

        This method assumes the provided move is valid.
        """
        if DEBUG_MODE:
            if not self.valid_moves_array()[move_index]:
                raise "Playing an invalid move!"
            if not self.unused_pieces[self.player][MOVES["piece_indices"][move_index]]:
                raise "Playing a piece that has already been used!"

        # Update occupancies.
        self.occupancies[self.player] |= MOVES["new_occupieds"][move_index]  

        # Update unused pieces.
        self.unused_pieces[self.player][MOVES["piece_indices"][move_index]] = False

        # Rule out some moves.
        self.moves_ruled_out |= MOVES["moves_ruled_out_for_all"][move_index]
        self.moves_ruled_out[self.player] |= MOVES["moves_ruled_out_for_player"][move_index]

        # Enable new moves for player based on corners.
        self.moves_enabled[self.player] |= MOVES["moves_enabled_for_player"][move_index]

        # Compute scores.
        self.accumulated_scores[self.player] += MOVES["scores"][move_index]

        if DEBUG_MODE:
            self.last_move_index = move_index        
        
        # Find the next player who has a valid move.
        for _ in range(4):
            self.player = (self.player + 1) % 4
            if self.valid_moves_array().any():
                return False
            
        return True

    def valid_moves_array(self):
        return ~self.moves_ruled_out[self.player] & self.moves_enabled[self.player]
    
    def result(self):
        r = np.where(self.accumulated_scores == np.max(self.accumulated_scores), 1, 0)
        return r / np.sum(r)
    
    def pretty_print_board(self):
        assert DEBUG_MODE, "Must have debug_mode set to print board."
        Display(self.occupancies, MOVES["new_occupieds"][self.last_move_index]).show()
