import time
import numpy as np

import player_pov_helpers
from state import State


BUFFER_LIMIT = 1000


class DataRecorder:
    def __init__(self, directory):
        self.directory = directory

        self.occupancies = []
        self.children_visit_distributions = []
        self.values = []

        # This is used to rotate the values correctly for each state when the game is over.
        self.players_on_current_game = []

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
    
    def record_game_end(self, values: np.ndarray):
        for player in self.players_on_current_game:
            self.values.append(player_pov_helpers.values_to_player_pov(values, player))

        self.players_on_current_game = []
        print("Occupancy size:", len(self.occupancies))
        if len(self.occupancies) > BUFFER_LIMIT:
            self.flush()

    def flush(self):
        if not self.occupancies:
            return

        key = str(int(time.time() * 1000))
        np.save(f"{self.directory}/occupancies_{key}.npy", np.array(self.occupancies))
        np.save(f"{self.directory}/children_visit_distributions_{key}.npy", np.array(self.children_visit_distributions))
        np.save(f"{self.directory}/values_{key}.npy", np.array(self.values))

        self.occupancies = []
        self.children_visit_distributions = []
        self.values = []