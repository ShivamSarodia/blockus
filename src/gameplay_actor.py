import ray
import asyncio
import random
import time
import pyinstrument

from configuration import config
from data_recorder import DataRecorder
from inference.client import InferenceClient
from mcts import MCTSAgent
from state import State


PROFILER_DIRECTORY = config()["development"]["profiler_directory"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]


@ray.remote
class GameplayActor:
    def __init__(self, inference_client: InferenceClient, output_data_dir: str):
        self.inference_client = inference_client
        self.data_recorder = DataRecorder(output_data_dir)

    async def run(self):
        print("Running gameplay process...")

        try:
            if PROFILER_DIRECTORY:
                profiler = pyinstrument.Profiler()
                profiler.start()
            await self.multi_continuously_play_games(COROUTINES_PER_PROCESS)
        except:
            self.data_recorder.flush()
            if PROFILER_DIRECTORY:
                profiler.stop()
                path = f"{PROFILER_DIRECTORY}/{int(time.time() * 1000)}_{random.getrandbits(30)}_gameplay.html"
                print(f"Writing profiler info to path: {path}")
                profiler.write_html(path)
            raise

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
            print(f"{time.time()}: Made move")

        self.data_recorder.record_game_end(recorder_game_id, state.result())


    async def continuously_play_games(self):
        while True:
            print(f"{time.time()}: Playing game...")
            start = time.time()
            await self.play_game()
            print(f"{time.time()}: Game finished in {time.time() - start:.2f}s")


    async def multi_continuously_play_games(self, num_coroutines: int):
        # We need to call this in here so that uvloop has had a chance to set the event loop first.
        self.inference_client.init_in_process(asyncio.get_event_loop())

        await asyncio.gather(
            *[self.continuously_play_games() for _ in range(num_coroutines)]
        )