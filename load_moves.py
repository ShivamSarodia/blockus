import os
import numpy as np

def load_moves():
    # For every file in precomputed_moves directory ending in .npy,
    # read the file into dictionary.
    moves = {}
    dir = "precomputed_moves/"
    filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    moves = {}
    for filename in filenames:
        if filename.endswith(".npy"):
            key = filename[:-4]
            with open(f"precomputed_moves/{filename}", "rb") as f:
                print("Loading file:", key)
                moves[key] = np.load(f)

    return moves