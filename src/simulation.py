import numpy as np
import multiprocessing

from configuration import config
from neural_net import NeuralNet
from inference.interface import InferenceInterface
from inference.evaluation import EvaluationEngine
from gameplay_engine import GameplayEngine
from inference.array_queue import ArrayQueue

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]
GAMEPLAY_PROCESSES = config()["architecture"]["gameplay_processes"]
EVALUATION_ENGINES_ON_GPU = config()["architecture"]["evaluation_engines_on_gpu"]
EVALUATION_ENGINERS_ON_CPU = config()["architecture"]["evaluation_engines_on_cpu"]


def run(output_data_dir):
    # First, initiate all the queues we'll need.

    # We need one request queue for storing all evaluation requests.
    # The capacity of this queue should be enough to store all requests that could be made at once,
    # which is the number of coroutines per process times the number of gameplay processes. We multiply
    # by two to ensure we don't run into any issues with the queue being full, although that shouldn't 
    # actually happen.
    request_queue_capacity = COROUTINES_PER_PROCESS * GAMEPLAY_PROCESSES * 2
    request_queue_item_likes = [
        np.empty((4, BOARD_SIZE, BOARD_SIZE), dtype=bool),
        np.empty((), dtype=int),
        np.empty((), dtype=int),
    ]
    request_queue = ArrayQueue(request_queue_capacity, request_queue_item_likes)

    # We need one output queue for each process that's doing evaluations. The capacity of each queue should be
    # enough to store all the results that could be produced by the process before they are read. We multiply
    # by two for the same reason as above.
    output_queues_map = {
        process_id: ArrayQueue(COROUTINES_PER_PROCESS * 2, [
            # Values
            np.empty((4,), dtype=float),
            # Policies
            np.empty((NUM_MOVES,), dtype=float),
            # Task IDs
            np.empty((), dtype=int),
        ])
        for process_id in range(GAMEPLAY_PROCESSES)
    }

    # Now, start all our processes.
    for i in range(EVALUATION_ENGINES_ON_GPU):
        evaluation_engine = EvaluationEngine(NeuralNet(), "mps", request_queue, output_queues_map)
        p = multiprocessing.Process(target=evaluation_engine.run, name=f"evaluation_engine_gpu_{i}")
        p.start()

    # TODO: Be more DRY.
    for i in range(EVALUATION_ENGINERS_ON_CPU):
        evaluation_engine = EvaluationEngine(NeuralNet(), "cpu", request_queue, output_queues_map)
        p = multiprocessing.Process(target=evaluation_engine.run, name=f"evaluation_engine_cpu_{i}")
        p.start()

    for process_id in range(GAMEPLAY_PROCESSES):
        inference_interface = InferenceInterface(process_id, request_queue, output_queues_map[process_id])
        gameplay_engine = GameplayEngine(inference_interface, output_data_dir)
        p = multiprocessing.Process(target=gameplay_engine.run, name=f"gameplay_{process_id}")
        p.start()

    # Join one of the gameplay processes so this lives forever.
    try:
        p.join()
    finally:
        print("Cleaning up.")
        request_queue.cleanup()
        for output_queue in output_queues_map.values():
            output_queue.cleanup()
