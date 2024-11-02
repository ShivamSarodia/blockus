import numpy as np
import multiprocessing

from configuration import config
from neural_net import NeuralNet
from inference.client import InferenceClient
from inference.actor import InferenceActor
from gameplay_engine import GameplayEngine


BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]


def run(output_data_dir): 
    # Start the Ray actor that runs GPU computations.
    inference_actor = InferenceActor.remote()
    inference_client = InferenceClient(inference_actor)

    # Now, this process will start to run gameplay.
    gameplay_engine = GameplayEngine(inference_client, output_data_dir)
    gameplay_engine.run()