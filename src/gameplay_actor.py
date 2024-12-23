import ray
import asyncio
import random
import time
import os
import copy
import pyinstrument
import json
from typing import Dict, Optional

from configuration import config, merge_into_dict
from data_recorder import DataRecorder
from inference.client import InferenceClient
from agents import PolicySamplingAgent, RandomAgent
from mcts import MCTSAgent
from state import State
from event_logger import log_event


USE_PROFILER = config()["development"]["profile"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]
AGENTS = config()["agents"]
LOG_MADE_MOVE = config()["logging"]["made_move"]

def generate_agent(
    agent_config: Dict,
    inference_clients: Dict[str, InferenceClient],
    data_recorder: Optional[DataRecorder],
    recorder_game_id: Optional[int],
):
    if agent_config["type"] == "mcts":
        network_name = agent_config["network"]
        return MCTSAgent(
            agent_config,
            inference_clients[network_name],
            data_recorder,
            recorder_game_id,
        )
    elif agent_config["type"] == "random":
        return RandomAgent()
    elif agent_config["type"] == "human":
        return None
    elif agent_config["type"] == "policy_sampling":
        network_name = agent_config["network"]
        return PolicySamplingAgent(
            agent_config,
            inference_clients[network_name],
        )
    else:
        raise "Unknown agent type."

@ray.remote
class GameplayActor:
    def __init__(self, actor_index: int, inference_clients: Dict[str, InferenceClient], output_data_dir: str):
        self.inference_clients = inference_clients
        self.output_data_dir = output_data_dir
        self.data_recorder = DataRecorder(output_data_dir)
        self.profiler = None

        self.agent_configs = [
            agent for agent in AGENTS
            if (
                "gameplay_actors" not in agent
                or
                actor_index in agent.get("gameplay_actors")
            )
        ]

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

        # Select four agents (without replacement)
        if len(self.agent_configs) == 1:
            agent_configs = [self.agent_configs[0]] * 4
        else:
            agent_configs = random.sample(self.agent_configs, k=4)

        agents = [
            generate_agent(
                agent_config,
                self.inference_clients,
                self.data_recorder,
                recorder_game_id,
            )
            for agent_config in agent_configs
        ]

        game_over = False
        state = State()
        shared_data = None
        while not game_over:
            agent = agents[state.player]
            move_index, shared_data = await agent.select_move_index(state, shared_data)
            game_over = state.play_move(move_index)
            if LOG_MADE_MOVE:
                log_event("made_move")
        
        result = state.result()
        log_event("game_result", 
            {
                "scores": [
                    [agent_configs[i]["name"], state.result()[i]]
                    for i in range(4)
                ],
                "game_id": recorder_game_id,
            }
        )

        self.data_recorder.record_game_end(recorder_game_id, result)

    async def continuously_play_games(self):
        while True:
            log_event("game_start")
            start = time.time()
            await self.play_game()
            log_event("game_end", { "runtime": time.time() - start })

    async def multi_continuously_play_games(self, num_coroutines: int):
        # We need to call this in here so that uvloop has had a chance to set the event loop first.
        for client in self.inference_clients.values():
            client.init_in_process(asyncio.get_event_loop())

        await asyncio.gather(
            *[self.continuously_play_games() for _ in range(num_coroutines)]
        )