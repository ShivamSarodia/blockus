from zipfile import BadZipFile
import numpy as np
from tqdm import tqdm
from event_logger import log_event

def load_games(game_file_paths):
    boards = []
    policies = []
    values = []
    # valid_moves = []
    # unused_pieces = []
    
    for game_file in game_file_paths:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                boards.append(npz["boards"])
                policies.append(npz["policies"])
                values.append(npz["values"])
                # valid_moves.append(npz["valid_moves_array"])
                # unused_pieces.append(npz["unused_pieces"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    if len(boards) == 0:
        return None

    return (
        np.concatenate(boards),
        np.concatenate(policies),
        np.concatenate(values),
        # np.concatenate(valid_moves),
        # np.concatenate(unused_pieces),
    )

def load_games_new(game_file_paths, with_tqdm=False):
    game_ids = []
    boards = []
    policies = []
    values = []
    valid_moves = []
    unused_pieces = []

    iterator = tqdm(game_file_paths) if with_tqdm else game_file_paths
    
    for game_file in iterator:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                boards.append(npz["boards"])
                policies.append(npz["policies"])
                values.append(npz["values"])
                if "game_ids" in npz:
                    game_ids.append(npz["game_ids"])
                valid_moves.append(npz["valid_moves_array"])
                unused_pieces.append(npz["unused_pieces"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    if len(boards) == 0:
        return None
    
    result = {
        "boards": np.concatenate(boards),
        "policies": np.concatenate(policies),
        "values": np.concatenate(values),
        "valid_moves": np.concatenate(valid_moves),
        "unused_pieces": np.concatenate(unused_pieces),
    }

    if game_ids:
        result["game_ids"] = np.concatenate(game_ids)

    return result
