import numpy as np
import random
from typing import Dict, NamedTuple
from tqdm import tqdm
import time
import asyncio
from multiprocessing import Pool


from config import config
from data_recorder import DataRecorder
from neural_net import NeuralNet
from inference import InferenceEngine
from mcts import MCTSAgent
from state import State


async def play_game(inference_engine: InferenceEngine, data_recorder: DataRecorder):
    recorder_game_id = data_recorder.start_game()

    agents = [
        MCTSAgent(inference_engine, data_recorder, recorder_game_id),
        MCTSAgent(inference_engine, data_recorder, recorder_game_id),
        MCTSAgent(inference_engine, data_recorder, recorder_game_id),
        MCTSAgent(inference_engine, data_recorder, recorder_game_id),
    ]

    game_over = False
    state = State()
    while not game_over:
        agent = agents[state.player]
        move_index = await agent.select_move_index(state)
        game_over = state.play_move(move_index)

    data_recorder.record_game_end(recorder_game_id, state.result())


async def continuously_play_games(inference_engine, data_recorder):
    while True:
        print("Playing game...")
        start = time.time()
        await play_game(inference_engine, data_recorder)
        print(f"Game finished in {time.time() - start:.2f}s")


async def multi_continuously_play_games(num_coroutines, inference_engine, data_recorder):
    await asyncio.gather(
        *[continuously_play_games(inference_engine, data_recorder) for _ in range(num_coroutines)]
    )


def run(output_data_dir):
    data_recorder = DataRecorder(output_data_dir)

    inference_engine = InferenceEngine()
    inference_engine.initialize()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(multi_continuously_play_games(20, inference_engine, data_recorder))
    except:
        data_recorder.flush()
        raise
