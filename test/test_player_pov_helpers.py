import numpy as np

import player_pov_helpers
from state import State
from constants import NUM_MOVES, BOARD_SIZE, MOVES

NUM_STEPS = 10

state = State()
for _ in range(NUM_STEPS):
    move_index = state.select_random_valid_move_index()

    pov_occupancies = player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)

    selected_move_array = np.zeros((NUM_MOVES,))
    selected_move_array[move_index] = 1
    pov_move_index = np.argmax(player_pov_helpers.moves_array_to_player_pov(selected_move_array, state.player))

    pov_occupancies[0] |= MOVES["new_occupieds"][pov_move_index]

    recreated_occupancies = player_pov_helpers.occupancies_to_player_pov(pov_occupancies, -state.player)

    state.play_move(move_index)
    assert np.all(state.occupancies == recreated_occupancies)

moves_array = np.random.random_sample((NUM_MOVES,))
for i in range(4):
    to_player_pov = player_pov_helpers.moves_array_to_player_pov(moves_array, i)
    back_to_universal = player_pov_helpers.moves_array_to_player_pov(to_player_pov, -i)

    assert np.all(np.equal(moves_array, back_to_universal))
