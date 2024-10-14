import time
import os
import numpy as np
import pickle

import player_pov_helpers
from constants import DEBUG_MODE
from state import State


BUFFER_LIMIT = 1000


class DataRecorder:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)        

        self.occupancies = []
        self.children_visit_distributions = []
        self.values = []

        # This is used to rotate the values correctly for each state when the game is over.
        self.players_on_current_game = []

        if DEBUG_MODE:
            self.states = []

    def record_rollout_result(self, state: State, children_visit_distribution: np.ndarray):
        """
        After each MCTS rollout completes, record the occupancy state did the rollout on and the final
        visit distribution of the children of that node.
        """
        self.occupancies.append(
            player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)
        )
        self.children_visit_distributions.append(
            player_pov_helpers.moves_array_to_player_pov(children_visit_distribution, state.player)
        )
        self.players_on_current_game.append(state.player)

        if DEBUG_MODE:
            self.states.append(state.clone())
    
    def record_game_end(self, values: np.ndarray):
        for player in self.players_on_current_game:
            self.values.append(player_pov_helpers.values_to_player_pov(values, player))

        self.players_on_current_game = []
        if len(self.occupancies) > BUFFER_LIMIT:
            self.flush()

    def flush(self):
        # Truncate the latest game, if it has not finished.
        if len(self.values) < len(self.occupancies):
            self.occupancies = self.occupancies[:len(self.values)]
            self.children_visit_distributions = self.children_visit_distributions[:len(self.values)]

        if not self.occupancies:
            return

        key = str(int(time.time() * 1000))
        np.save(f"{self.directory}/occupancies_{key}.npy", np.array(self.occupancies))
        np.save(f"{self.directory}/children_visit_distributions_{key}.npy", np.array(self.children_visit_distributions))
        np.save(f"{self.directory}/values_{key}.npy", np.array(self.values))

        self.occupancies = []
        self.children_visit_distributions = []
        self.values = []

        if DEBUG_MODE:
            with open(f"{self.directory}/states_{key}.pkl", "wb") as f:
                pickle.dump(self.states, f)
            self.states = []