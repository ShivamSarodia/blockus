import ray
import os
import time
import torch
from datetime import datetime
from typing import Dict
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from neural_net import NeuralNet
from event_logger import log_event
from configuration import config 
from training.load_games import load_games

TRAINING_RUN = config()["training"]["run"]
if TRAINING_RUN:
    NETWORK_NAME = config()["training"]["network_name"]
    NETWORK_CONFIG = config()["networks"][NETWORK_NAME]
    NEW_DATA_CHECK_INTERVAL = config()["training"]["new_data_check_interval"]
    BATCH_SIZE = config()["training"]["batch_size"]
    POLICY_LOSS_WEIGHT = config()["training"]["policy_loss_weight"]
    LEARNING_RATE = config()["training"]["learning_rate"]

@ray.remote
class TrainingActor:
    def __init__(self, gamedata_path) -> None:
        self.gamedata_path = gamedata_path
        self.last_read_game_file_path = ""

    def run(self):
        # Double check this actor is even supposed to be running.
        assert TRAINING_RUN

        # First, limit training to just one CPU so we don't hog 
        # resources needed for self-play. 
        torch.set_num_threads(1)

        # Start by loading the base version of the model to train.
        # This is the latest model in the model directory.
        model = NeuralNet(NETWORK_CONFIG)
        model.load_state_dict(torch.load(self._find_latest_model_path(), weights_only=True))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Now, execute our training loop.
        while True:
            time.sleep(NEW_DATA_CHECK_INTERVAL)

            new_game_paths = self._get_unread_game_data_paths()
            if not new_game_paths:
                log_event("training_skip")
                continue

            model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

            # Load the new game data.
            boards, policies, values = load_games(new_game_paths)

            log_event("training_start", {
                "model_name": model_name,
                "num_samples": len(boards),
            })

            dataset = TensorDataset(
                torch.Tensor(boards),
                torch.Tensor(policies),
                torch.Tensor(values)
            )
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Train the network on the new data.
            for i, (boards, policies, values) in enumerate(dataloader):
                pred_values, pred_policy = model(boards)

                value_loss = nn.CrossEntropyLoss()(
                    pred_values,
                    values,
                )
                policy_loss = POLICY_LOSS_WEIGHT * nn.CrossEntropyLoss()(
                    pred_policy,
                    policies,
                )
                loss = value_loss + policy_loss

                log_event("training_batch", {
                    "model_name": model_name,
                    "iteration": i,
                    "value_loss": value_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "loss": loss.item(),
                })

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save the model to the model directory.
            model_path = os.path.join(
                NETWORK_CONFIG["model_directory"],
                f"{model_name}.pt",
            )
            torch.save(model.state_dict(), model_path)

            log_event("training_complete", {
                model_name: model_name,
            })

    def _get_unread_game_data_paths(self):
        gamedata_paths = [
            os.path.join(self.gamedata_path, filename)
            for filename in os.listdir(self.gamedata_path)
            if filename.endswith(".npz")
        ]

        # Select the game data files that haven't been read yet.
        unread_paths = [
            path for path in gamedata_paths
            if path > self.last_read_game_file_path
        ]

        if unread_paths:
            self.last_read_game_file_path = max(unread_paths)

        return unread_paths

    def _find_latest_model_path(self):
        # TODO: DRY this with the same method in inference/actor.py
        model_dir = NETWORK_CONFIG["model_directory"]
        model_paths = [
            os.path.join(model_dir, filename)
            for filename in os.listdir(model_dir)
            if (
                os.path.isfile(os.path.join(model_dir, filename)) and 
                filename.endswith(".pt")
            )
        ]
        return max(model_paths)