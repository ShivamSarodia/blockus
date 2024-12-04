import ray
import os
import time
import torch
from typing import Dict

from neural_net import NeuralNet
from event_logger import log_event
from configuration import config 
from training.game_data_manager import DirectoryGameDataPathFetcher, GameDataManager
import training.helpers

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
    MINIMUM_WINDOW_SIZE = config()["training"]["minimum_window_size"]

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
        latest_model_path = self._find_latest_model_path()
        latest_model_sample_count = int(latest_model_path.split("/")[-1].split(".")[0])

        model = NeuralNet(NETWORK_CONFIG)
        model.load_state_dict(torch.load(latest_model_path, weights_only=True))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        game_data_path_fetcher = DirectoryGameDataPathFetcher(self.gamedata_path)
        game_data_manager = GameDataManager(game_data_path_fetcher, SAMPLE_WINDOW)

        # As this method is currently implemented, this doesn't work
        # well at all when the latest_model_sample_count is large because 
        # it reads in all the data files into memory before dropping the 
        # old ones. For a latest_model_sample_count of 600k, this takes
        # like 30 seconds. I think I want to fix this by modifying the 
        # behavior of feed_window if it's asked to read in more samples
        # than the window size to intelligently skip over the early samples
        # that will be excluded.
        log_event("training_feeding_window", {
            "latest_model_sample_count": latest_model_sample_count,
            "minimum_window_size": MINIMUM_WINDOW_SIZE,
            "amount_to_feed": max(latest_model_sample_count, MINIMUM_WINDOW_SIZE),
        })
        training.helpers.feed_window_until_amount(
            game_data_manager,
            max(latest_model_sample_count, MINIMUM_WINDOW_SIZE),
            NEW_DATA_CHECK_INTERVAL,
        )
        log_event("training_window_fed", {})

        # Now, we're ready to start training.
        previous_cumulative_window_fed = game_data_manager.cumulative_window_fed()
        while True:
            training_result = training.helpers.loop_iteration(
                model,
                optimizer,
                game_data_manager,
                device="cpu",
                batch_size=BATCH_SIZE,
                sampling_ratio=SAMPLING_RATIO,
                policy_loss_weight=POLICY_LOSS_WEIGHT,
            )
            if not training_result:
                log_event("training_skip", {})
                time.sleep(NEW_DATA_CHECK_INTERVAL)
                continue

            log_event("training_batch", training_result)

            # Potentially save the model.
            cumulative_window_fed = training_result["cumulative_window_fed"]
            if cumulative_window_fed // SAMPLES_PER_GENERATION > previous_cumulative_window_fed // SAMPLES_PER_GENERATION:
                model_name = str(cumulative_window_fed).zfill(9)
                model_path = os.path.join(
                    NETWORK_CONFIG["model_directory"],
                    f"{model_name}.pt",
                )
                torch.save(model.state_dict(), model_path)
                log_event("saved_model", {
                    "cumulative_window_fed": cumulative_window_fed,
                    "model_name": model_name,
                    "window_size": game_data_manager.current_window_size(),
                })
            previous_cumulative_window_fed = cumulative_window_fed

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