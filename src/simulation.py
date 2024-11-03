import numpy as np
import multiprocessing
import ray

from configuration import config
from neural_net import NeuralNet
from inference.client import InferenceClient
from inference.actor import InferenceActor
from gameplay_actor import GameplayActor


BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
GAMEPLAY_PROCESSES = config()["architecture"]["gameplay_processes"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]


def run(output_data_dir): 
    # Start the Ray actor that runs GPU computations.
    inference_actor = InferenceActor.remote()
    inference_client = InferenceClient(inference_actor)

    # Now, start two Ray actors that run gameplay.
    gameplay_actors = [
        GameplayActor.remote(inference_client, output_data_dir)
        for _ in range(GAMEPLAY_PROCESSES)
    ]

    # Blocks indefinitely, because gameplay actor never finishes.
    ray.get([
        gameplay_actor.run.remote() for gameplay_actor in gameplay_actors
    ])