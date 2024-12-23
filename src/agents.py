import numpy as np
import json

import player_pov_helpers
from inference.client import InferenceClient
from state import State
from configuration import config, moves_data

BOARD_SIZE = config()["game"]["board_size"]
MOVES_DATA = moves_data()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RandomAgent:
    async def select_move_index(self, state: State, shared_data: dict):
        # Get all valid moves.
        valid_moves = np.flatnonzero(state.valid_moves_array())
        
        # Find the move with the highest number of newly occupied squares.
        best_move_score = -1
        for move_index in valid_moves:
            best_move_score = max(best_move_score, np.sum(MOVES_DATA["new_occupieds"][move_index]))

        # Select a move randomly from all the best moves.
        best_moves = []
        for move_index in valid_moves:
            if np.sum(MOVES_DATA["new_occupieds"][move_index]) == best_move_score:
                best_moves.append(move_index)

        return np.random.choice(best_moves), None
    

class PolicySamplingAgent:
    def __init__(self, agent_config, inference_client: InferenceClient):
        self.agent_config = agent_config
        self.inference_client = inference_client

    async def select_move_index(self, state: State, shared_data: dict):
        array_index_to_move_index = np.flatnonzero(state.valid_moves_array())

        player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(
            state.occupancies,
            state.player,
        )
        player_pov_valid_move_indices = player_pov_helpers.moves_indices_to_player_pov(
            array_index_to_move_index,
            state.player,
        )
        
        _, universal_children_prior_logits = await self.inference_client.evaluate(
            player_pov_occupancies,
            player_pov_valid_move_indices,
            state.turn,
        )
        universal_children_priors = softmax(universal_children_prior_logits)

        temperature = self.agent_config["move_selection_temperature"]
        if temperature == 0:
            return array_index_to_move_index[np.argmax(universal_children_priors)], None
        else:
            weighted_probabilities = np.power(
                universal_children_priors,
                1 / temperature,
            )
            probabilities = weighted_probabilities / np.sum(weighted_probabilities)
            return np.random.choice(array_index_to_move_index, p=probabilities), None
