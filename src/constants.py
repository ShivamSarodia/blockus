import os
import numpy as np

MOVES = None
NUM_MOVES = None
BOARD_SIZE = None
DEBUG_MODE = False


def load(moves_dir, debug_mode):
    global MOVES
    global NUM_MOVES
    global BOARD_SIZE
    global DEBUG_MODE

    DEBUG_MODE = debug_mode

    # For every file in precomputed_moves directory ending in .npy,
    # read the file into dictionary.
    moves = {}
    filenames = [f for f in os.listdir(moves_dir) if os.path.isfile(os.path.join(moves_dir, f))]

    moves = {}
    for filename in filenames:
        if filename.endswith(".npy"):
            key = filename[:-4]
            with open(f"{moves_dir}/{filename}", "rb") as f:
                print("Loading file:", key)
                moves[key] = np.load(f)

    MOVES = moves

    NUM_MOVES = MOVES["new_occupieds"].shape[0]   
    print("Number of moves:", NUM_MOVES)

    BOARD_SIZE = MOVES["new_occupieds"].shape[1]
    print("Board size:", BOARD_SIZE)

    return MOVES
