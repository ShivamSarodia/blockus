import os
import numpy as np

MOVES = None
NUM_MOVES = None
BOARD_SIZE = None


def load_moves(dir):
    global MOVES
    global NUM_MOVES
    global BOARD_SIZE

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

    MOVES = moves

    NUM_MOVES = MOVES["new_occupieds"].shape[0]   
    print("Number of moves:", NUM_MOVES)

    BOARD_SIZE = MOVES["new_occupieds"].shape[1]
    print("Board size:", BOARD_SIZE)

    return MOVES

# TODO: Control this by command line instead.
load_moves("data/moves_10")