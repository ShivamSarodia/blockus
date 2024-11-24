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
    SAMPLE_WINDOW = config()["training"]["sample_window"]
    SAMPLES_PER_GENERATION = config()["training"]["samples_per_generation"]
    SAMPLING_RATIO = config()["training"]["sampling_ratio"]
    SAMPLING_DELAY = config()["training"]["sampling_delay"]

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

        model_path = self._find_latest_model_path()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # The model path contains the number of samples we've read so far.
        model_samples_read = int(model_path.split("/")[-1].split(".")[0])

        if model_samples_read > 0:
            samples_read = model_samples_read
        else:
            samples_read = SAMPLING_DELAY

        while True:
            # Check how many samples are available.
            gamedata_paths = [
                os.path.join(self.gamedata_path, filename)
                for filename in os.listdir(self.gamedata_path)
                if filename.endswith(".npz")
            ]
            gamedata_paths.sort()
            sample_counts = [
                int(path.split("_")[-1].split(".")[0]) for path in gamedata_paths
            ]
            total_samples_available_on_disk = sum(sample_counts)

            # To train a new generation, we need to have enough new samples that we can
            # afford to train on SAMPLES_PER_GENERATION samples without exceeding the sampling
            # ratio.
            # 
            # For example, suppose each generation trains on 10,000 samples and our sampling 
            # ratio is 2. Then, each time there is 5,000 new samples available, we can train 
            # a new generation.
            new_samples_needed = SAMPLES_PER_GENERATION // SAMPLING_RATIO
            if total_samples_available_on_disk - samples_read < new_samples_needed:
                log_event("training_skip", {
                    "total_samples_available_on_disk": total_samples_available_on_disk,
                    "samples_read": samples_read,
                    "new_samples_needed": new_samples_needed,
                })
                time.sleep(NEW_DATA_CHECK_INTERVAL)
                continue
            
            log_event("training_start", {
                "total_samples_available_on_disk": total_samples_available_on_disk,
                "samples_read": samples_read,
                "new_samples_needed": new_samples_needed,                    
            })

            # OK, now we know there's enough new samples that we can afford to train a new
            # generation.
            # 
            # Technically, this is a bit incorrect -- if more than new_samples_needed samples 
            # appear in one iteration, we would ideally train two models in order to prevent
            # from undersampling. However, I think it should be unlikely that this happens.
            # 
            # We can watch for this by looking for instances of training_start where the 
            # total_samples_available_on_disk - samples_read is >= 2 * new_samples_needed.
            # 
            # TODO: We could change this a bit -- either maintain an active "sample budget"
            # where excess samples are saved to use later, or maybe publish metrics somewhere
            # that help us monitor the _real_ sampling ratio.
            samples_read = total_samples_available_on_disk

            # Walking backwards from the latest file, accumulate a list of all files such that
            # the total number of samples in the window is no more than SAMPLE_WINDOW.
            i = len(sample_counts) - 1
            samples_in_window = 0
            gamedata_paths_in_window = []
            while i >= 0 and samples_in_window < SAMPLE_WINDOW:
                gamedata_paths_in_window.append(gamedata_paths[i])
                samples_in_window += sample_counts[i]
                i -= 1

            # The model name is just the number of samples we're training this guy on.
            model_name = str(samples_read).zfill(9)

            # Load the new game data.
            boards, policies, final_game_values, average_rollout_values = load_games(gamedata_paths_in_window)

            # TODO: Make this ratio a configuration.
            values = (final_game_values + average_rollout_values) / 2

            log_event("training_window_loaded", {
                "model_name": model_name,
                "samples_read": samples_read,
                "samples_in_window": len(boards),
            })

            dataset = TensorDataset(
                torch.Tensor(boards),
                torch.Tensor(policies),
                torch.Tensor(values)
            )
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Train the network on the new data.
            samples_trained_on = 0
            for i, (boards, policies, values) in enumerate(dataloader):

                # Track how many samples we've trained on so that we can stop once we've sampled
                # enough for a new generation.
                if samples_trained_on >= SAMPLES_PER_GENERATION:
                    break
                samples_trained_on += len(boards)

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
                    "batch_size": len(boards),
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
                samples_trained_on: samples_trained_on,
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
        return max(model_paths)#