import os
import numpy as np
import tomllib
import neptune as neptune_module

_CONFIG = None
_MOVES = None

def _load_config():
    global _CONFIG
    with open(os.environ["CONFIG_PATH"], "rb") as f:
        _CONFIG = tomllib.load(f)
    print("Loaded config from file:", os.environ["CONFIG_PATH"])

def config():
    global _CONFIG

    if _CONFIG is None:
        _load_config()

    return _CONFIG

def moves_data():
    global _MOVES

    if _MOVES is None:
        _load_moves()

    return _MOVES

def _load_moves():
    global _MOVES

    moves_dir = config()["game"]["moves_directory"]
    filenames = [f for f in os.listdir(moves_dir) if os.path.isfile(os.path.join(moves_dir, f))]

    moves = {}
    for filename in filenames:
        if filename.endswith(".npy"):
            key = filename[:-4]
            with open(f"{moves_dir}/{filename}", "rb") as f:
                print("Loading file:", key)
                moves[key] = np.load(f)

    _MOVES = moves

    assert _MOVES["new_occupieds"].shape[0] == config()["game"]["num_moves"]
    assert _MOVES["new_occupieds"].shape[1] == config()["game"]["board_size"]
