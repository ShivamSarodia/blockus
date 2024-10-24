import numpy as np
import random
from typing import Dict, NamedTuple
from tqdm import tqdm
import time
import asyncio
import uvloop
import multiprocessing
import pyinstrument
from multiprocessing import shared_memory

from configuration import config
from data_recorder import DataRecorder
from neural_net import NeuralNet
from inference.interface import InferenceInterface
from inference.evaluation import EvaluationEngine
from mcts import MCTSAgent
from inference.array_queue import ArrayQueue
from state import State

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
PROFILER_DIRECTORY = config()["development"]["profiler_directory"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]
GAMEPLAY_PROCESSES = config()["architecture"]["gameplay_processes"]
EVALUATION_ENGINES_ON_GPU = config()["architecture"]["evaluation_engines_on_gpu"]
EVALUATION_ENGINERS_ON_CPU = config()["architecture"]["evaluation_engines_on_cpu"]


async def play_game(inference_interface: InferenceInterface, data_recorder: DataRecorder):
    recorder_game_id = data_recorder.start_game()

    agents = [
        MCTSAgent(inference_interface, data_recorder, recorder_game_id),
        MCTSAgent(inference_interface, data_recorder, recorder_game_id),
        MCTSAgent(inference_interface, data_recorder, recorder_game_id),
        MCTSAgent(inference_interface, data_recorder, recorder_game_id),
    ]

    game_over = False
    state = State()
    while not game_over:
        agent = agents[state.player]
        move_index = await agent.select_move_index(state)
        game_over = state.play_move(move_index)
        print(f"{time.time()}: Made move")

    data_recorder.record_game_end(recorder_game_id, state.result())


async def continuously_play_games(inference_interface: InferenceInterface, data_recorder):
    while True:
        print(f"{time.time()}: Playing game...")
        start = time.time()
        await play_game(inference_interface, data_recorder)
        print(f"{time.time()}: Game finished in {time.time() - start:.2f}s")


async def multi_continuously_play_games(num_coroutines, inference_interface: InferenceInterface, data_recorder):
    await asyncio.gather(
        *[continuously_play_games(inference_interface, data_recorder) for _ in range(num_coroutines)]
    )


def run(output_data_dir):
    # First, initiate all the queues we'll need.

    # We need one request queue for storing all evaluation requests.
    # The capacity of this queue should be enough to store all requests that could be made at once,
    # which is the number of coroutines per process times the number of gameplay processes. We multiply
    # by ten to ensure we don't run into any issues with the queue being full.
    request_queue_capacity = COROUTINES_PER_PROCESS * GAMEPLAY_PROCESSES * 10
    request_queue_item_likes = [
        np.empty((4, BOARD_SIZE, BOARD_SIZE), dtype=bool),
        np.empty((), dtype=int),
        np.empty((), dtype=int),
    ]
    request_queue = ArrayQueue(request_queue_capacity, request_queue_item_likes)

    # We need one output queue for each process that's doing evaluations. The capacity of each queue should be
    # enough to store all the results that could be produced by the process before they are read. We multiply
    # by ten for the same reason as above.
    output_queues_map = {
        process_id: ArrayQueue(COROUTINES_PER_PROCESS * 10, [
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
        p = multiprocessing.Process(target=_run_gameplay_process, args=(output_data_dir, inference_interface), name=f"gameplay_{process_id}")
        p.start()

    # Join one of the gameplay processes so this lives forever.
    try:
        p.join()
    finally:
        print("Cleaning up shared memory.")
        request_queue.cleanup()
        for output_queue in output_queues_map.values():
            output_queue.cleanup()

def _run_gameplay_process(output_data_dir, inference_interface: InferenceInterface):
    print("Running gameplay process...")

    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)

    data_recorder = DataRecorder(output_data_dir)
    inference_interface.init_in_process(loop)

    try:
        if PROFILER_DIRECTORY:
            profiler = pyinstrument.Profiler()
            profiler.start()
        loop.run_until_complete(multi_continuously_play_games(
            COROUTINES_PER_PROCESS,
            inference_interface,
            data_recorder,
        ))
    except:
        data_recorder.flush()
        profiler.stop()
        profiler.write_html(
            f"{PROFILER_DIRECTORY}/{int(time.time() * 1000)}_gameplay.html",
        )
        raise
