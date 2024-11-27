import time
import os
import numpy as np
import random

from configuration import config
import player_pov_helpers
from state import State

GAME_FLUSH_THRESHOLD = config()["architecture"]["game_flush_threshold"]

class DataRecorder:
    def __init__(self, directory):
        self.directory = os.path.join(directory, "games/")
        os.makedirs(self.directory, exist_ok=True)

        # This object is a map from a randomly generated game ID to a dictionary of the form:
        # {
        #     "occupancies": [],
        #     "policies": [],
        #     "players": [],
        #     "valid_moves_array": [],
        #     "average_rollout_values": [],
        #     "final_game_values": [],
        #     "game_ids": [],
        # }
        self.games = {}
        self.finished_games = set()

    def start_game(self) -> int:
        game_id = random.randint(0, (1<<31) - 1)
        self.games[game_id] = {
            "occupancies": [],
            "policies": [],
            "players": [],
            "valid_moves_array": [],
            "average_rollout_values": [],
            "final_game_values": [],
            "game_ids": [],
        }
        return game_id

    def record_rollout_result(self, game_id: int, state: State, policy: np.ndarray, average_rollout_value: np.ndarray):
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
        game["valid_moves_array"].append(
            player_pov_helpers.moves_array_to_player_pov(
                state.valid_moves_array(),
                state.player,
            )
        )
        game["average_rollout_values"].append(
            player_pov_helpers.values_to_player_pov(average_rollout_value, state.player)
        )
        game["players"].append(state.player)
        game["game_ids"].append(game_id)
    
    def record_game_end(self, game_id: int, values: np.ndarray):
        game = self.games[game_id]
        for player in game["players"]:
            game["final_game_values"].append(player_pov_helpers.values_to_player_pov(values, player))

        self.finished_games.add(game_id)
        if len(self.finished_games) >= GAME_FLUSH_THRESHOLD:
            self.flush()

    def flush(self):
        game_ids = []
        occupancies = []
        policies = []
        valid_moves_array = []
        average_rollout_values = []
        final_game_values = []

        for game_id in self.finished_games:
            game = self.games[game_id]

            # Some small number of games will have no data, because they were composed
            # only of fast rollouts that didn't report any states.
            if len(game["occupancies"]) > 0:
                occupancies.append(np.array(game["occupancies"]))
                policies.append(np.array(game["policies"]))
                valid_moves_array.append(np.array(game["valid_moves_array"]))
                final_game_values.append(np.array(game["final_game_values"]))
                average_rollout_values.append(np.array(game["average_rollout_values"]))
                game_ids.append(np.array(game["game_ids"]))

            del self.games[game_id]
        
        self.finished_games = set()

        if not game_ids:
            return
        
        game_ids = np.concatenate(game_ids)
        occupancies = np.concatenate(occupancies)
        policies = np.concatenate(policies)
        valid_moves_array = np.concatenate(valid_moves_array)
        final_game_values = np.concatenate(final_game_values)
        average_rollout_values = np.concatenate(average_rollout_values)

        # Save the files to disk with the number of samples included, so that the
        # training script can tell from just the filename how many samples are in
        # the file.
        np.savez_compressed(
            os.path.join(self.directory, f"{int(time.time() * 1000)}_{len(game_ids)}.npz"),
            game_ids=game_ids,
            occupancies=occupancies,
            policies=policies,
            final_game_values=final_game_values,
            average_rollout_values=average_rollout_values,
            valid_moves_array=valid_moves_array,
        )
