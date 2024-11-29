from zipfile import BadZipFile
import numpy as np
from event_logger import log_event


def load_games(game_file_paths):
    occupancies = []
    policies = []
    final_game_values = []
    average_rollout_values = []
    game_ids = []

    for game_file in game_file_paths:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                occupancies.append(npz["occupancies"])
                policies.append(npz["policies"])
                final_game_values.append(npz["final_game_values"])
                average_rollout_values.append(npz["average_rollout_values"])
                game_ids.append(npz["game_ids"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    if len(occupancies) == 0:
        return None

    return (
        np.concatenate(occupancies),
        np.concatenate(policies),
        np.concatenate(final_game_values),
        np.concatenate(average_rollout_values),
    )


def load_old_format_games(game_file_paths):
    occupancies = []
    policies = []
    values = []
    game_ids = []

    for game_file in game_file_paths:
        with open(game_file, "rb") as f:
            try:
                npz = np.load(f)
                occupancies.append(npz["occupancies"])
                policies.append(npz["policies"])
                values.append(npz["values"])
                game_ids.append(npz["game_ids"])
            except BadZipFile:
                log_event("bad_game_file", {"path": game_file})

    return (
        np.concatenate(occupancies),
        np.concatenate(policies),
        np.concatenate(values),
    )