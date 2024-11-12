import os
import torch
import numpy as np


def load_games(games_directory, test_fraction=0.1):
    game_file_paths = [
        os.path.join(games_directory, game_file)
        for game_file in os.listdir(games_directory)
    ]

    # TODO: Instead of splitting by file, load everything and then split by 
    # game ID?

    num_test_files = round(len(game_file_paths) * test_fraction)
    print(f"Reserving {num_test_files} files for testing.")

    train_data = _load_from_paths(game_file_paths[:-num_test_files])
    test_data = _load_from_paths(game_file_paths[-num_test_files:])

    print(f"Loaded {len(train_data[0])} training samples.")
    print(f"Loaded {len(test_data[0])} testing samples.")

    return train_data, test_data


def _load_from_paths(game_file_paths):
    occupancies = []
    policies = []
    values = []
    game_ids = []

    for game_file in game_file_paths:
        with open(game_file, "rb") as f:
            npz = np.load(f)
            occupancies.append(npz["occupancies"])
            policies.append(npz["policies"])
            values.append(npz["values"])
            game_ids.append(npz["game_ids"])

    return (
        torch.Tensor(np.concatenate(occupancies)),
        torch.Tensor(np.concatenate(policies)),
        torch.Tensor(np.concatenate(values)),
    )