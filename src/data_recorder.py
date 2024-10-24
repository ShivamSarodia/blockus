import time
import os
import numpy as np
import pickle
import random

import player_pov_helpers
from state import State

class DataRecorder:
    def __init__(self, directory, game_flush_threshold=10):
        self.game_flush_threshold = game_flush_threshold

        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        # This object is a map from a randomly generated game ID to a dictionary of the form:
        # {
        #     "occupancies": [],
        #     "policies": [],
        #     "players": [],
        #     "values": [],  (populated only if DEBUG MODE is true)
        # }
        self.games = {}
        self.finished_games = set()

    def start_game(self) -> int:
        game_id = random.randint(0, (1<<31) - 1)
        self.games[game_id] = {
            "occupancies": [],
            "policies": [],
            "players": [],
            "values": []
        }
        return game_id

    def record_rollout_result(self, game_id: int, state: State, policy: np.ndarray):
        """
        After each MCTS rollout completes, record the occupancy state did the rollout on and the final
        visit distribution of the children of that node.
        """
        game = self.games[game_id]
        game["occupancies"].append(
            player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)
        )
        game["policies"].append(
            player_pov_helpers.moves_array_to_player_pov(policy, state.player)
        )
        game["players"].append(state.player)
    
    def record_game_end(self, game_id: int, values: np.ndarray):
        game = self.games[game_id]
        for player in game["players"]:
            game["values"].append(player_pov_helpers.values_to_player_pov(values, player))
        self.finished_games.add(game_id)

        print("Finished a game!")

        if len(self.finished_games) > self.game_flush_threshold:
            self.flush()

    def flush(self):
        game_ids = []
        occupancies = []
        policies = []
        values = []

        for game_id in self.finished_games:
            game = self.games[game_id]

            occupancies.append(np.array(game["occupancies"]))
            policies.append(np.array(game["policies"]))
            values.append(np.array(game["values"]))
            game_ids.append(game_id)

            del self.games[game_id]
        
        self.finished_games = set()

        if not game_ids:
            print("No complete games recorded.")
            return

        np.savez(
            os.path.join(self.directory, f"{int(time.time() * 1000)}.npz"),
            game_ids=np.array(game_ids),
            occupancies=np.concatenate(occupancies),
            policies=np.concatenate(policies),
            values=np.concatenate(values)
        )
