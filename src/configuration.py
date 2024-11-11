import os
import numpy as np
import tomllib
import json

_CONFIG = None
_MOVES = None

def merge_into_dict(original_dict, new_values):
    assert isinstance(original_dict, dict)
    assert isinstance(new_values, dict)
    for key in new_values:
        if key in original_dict:
            if isinstance(original_dict[key], dict):
                merge_into_dict(original_dict[key], new_values[key])
            else:
                original_dict[key] = new_values[key]
        else:
            original_dict[key] = new_values[key]

def _load_config():
    global _CONFIG
    config = {}
    config_paths = os.environ["CONFIG_PATHS"].split(",")
    for config_path in config_paths:
        with open(config_path, "rb") as f:
            config_level = tomllib.load(f)
            merge_into_dict(config, config_level)
    print("Loaded config: ", json.dumps(config))
    _CONFIG = config

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
