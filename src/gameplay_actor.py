import ray
import asyncio
import random
import time
import os
import pyinstrument

from configuration import config
from data_recorder import DataRecorder
from inference.client import InferenceClient
from mcts import MCTSAgent
from state import State
from event_logger import log_event


USE_PROFILER = config()["development"]["profile"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]


@ray.remote
class GameplayActor:
    def __init__(self, inference_client: InferenceClient, output_data_dir: str):
        self.inference_client = inference_client
        self.output_data_dir = output_data_dir
        self.data_recorder = DataRecorder(output_data_dir)
        self.profiler = None

    async def run(self):
        print("Running gameplay process...")

        # Setup.
        if USE_PROFILER:
            self.profiler = pyinstrument.Profiler()
            self.profiler.start()

        # Play games.
        await self.multi_continuously_play_games(COROUTINES_PER_PROCESS)

    def cleanup(self):
        print("Cleaning up gameplay actor...")
        self.data_recorder.flush()
        if USE_PROFILER:
            self.profiler.stop()
            path = os.path.join(self.output_data_dir, f"profiler/")
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"{random.getrandbits(30)}_gameplay.html")
            print(f"Writing profiler info to path: {path}")
            self.profiler.write_html(path)

    async def play_game(self):
        recorder_game_id = self.data_recorder.start_game()

        agents = [
            MCTSAgent(self.inference_client, self.data_recorder, recorder_game_id),
            MCTSAgent(self.inference_client, self.data_recorder, recorder_game_id),
            MCTSAgent(self.inference_client, self.data_recorder, recorder_game_id),
            MCTSAgent(self.inference_client, self.data_recorder, recorder_game_id),
        ]

        game_over = False
        state = State()
        while not game_over:
            agent = agents[state.player]
            move_index = await agent.select_move_index(state)
            game_over = state.play_move(move_index)
            log_event("made_move")

        self.data_recorder.record_game_end(recorder_game_id, state.result())


    async def continuously_play_games(self):
        while True:
            log_event("game_start")
            start = time.time()
            await self.play_game()
            log_event("game_end", { "runtime": time.time() - start })

    async def multi_continuously_play_games(self, num_coroutines: int):
        # We need to call this in here so that uvloop has had a chance to set the event loop first.
        self.inference_client.init_in_process(asyncio.get_event_loop())

        await asyncio.gather(
            *[self.continuously_play_games() for _ in range(num_coroutines)]
        )