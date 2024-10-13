import numpy as np 
from moves import MOVES

# A negative value in these methods moves from that player's POV 
# back to universal POV.

def occupancies_to_player_pov(occupancies, player):
    rotated_board = np.rot90(occupancies, k=player, axes=(-2, -1))
    return np.roll(rotated_board, shift=-player, axis=-3) 


def moves_array_to_player_pov(moves_array, player):
    return moves_array[MOVES["rotation_mapping"][(-player) % 4]]


def values_to_player_pov(values, player):
    return np.roll(values, shift=-player, axis=-1)