from typing import Dict
import numpy as np

from configuration import config
from state import State
import player_pov_helpers
from data_recorder import DataRecorder
from inference.interface import InferenceInterface


NUM_MOVES = config()["game"]["num_moves"]
NUM_MCTS_ROLLOUTS = config()["mcts"]["num_rollouts"]
UCB_DEFAULT_CHILD_VALUE = config()["mcts"]["ucb_default_child_value"]
UCB_EXPLORATION = config()["mcts"]["ucb_exploration"]
ROOT_EXPLORATION_FRACTION = config()["mcts"]["root_exploration_fraction"]
ROOT_DIRICHLET_ALPHA = config()["mcts"]["root_dirichlet_alpha"]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MCTSAgent:
    def __init__(self, inference_interface: InferenceInterface, data_recorder: DataRecorder, recorder_game_id: int):
        self.inference_interface = inference_interface
        self.data_recorder = data_recorder
        self.recorder_game_id = recorder_game_id

    async def select_move_index(self, state: State):
        search_root = MCTSValuesNode()
        await search_root.get_value_and_expand_children(state, self.inference_interface)

        for _ in range(NUM_MCTS_ROLLOUTS):
            scratch_state = state.clone()

            # At the start of each iteration, nodes_visited will be one 
            # longer than moves_played. moves_played[i] is the move played
            # to exit nodes_visited[i].
            nodes_visited = [search_root]
            moves_played = []

            while True:
                move_index = nodes_visited[-1].select_child_by_ucb(scratch_state)
                moves_played.append(move_index)

                game_over = scratch_state.play_move(move_index)
                # print("Move played:", move_index)
                # state.pretty_print_board()

                # If the game is over, we can now assign values based on the final state.
                if game_over:
                    break

                next_node = nodes_visited[-1].move_index_to_child_node.get(move_index)

                # If next_node does not exist, we need to break out of this loop to create that node and
                # fetch a new value to backpropagate.
                if not next_node:
                    break

                nodes_visited.append(next_node)

            if game_over:
                value = scratch_state.result()
            else:
                new_node = MCTSValuesNode()
                value = await new_node.get_value_and_expand_children(scratch_state, self.inference_interface)
                nodes_visited[-1].move_index_to_child_node[moves_played[-1]] = new_node

            # Now, backpropagate the value up the visited notes.
            for i in range(len(nodes_visited)):
                node = nodes_visited[i]
                node_array_index = node.move_index_to_array_index[moves_played[i]]
                node.children_value_sums[:,node_array_index] += value 
                node.children_visit_counts[node_array_index] += 1

        # Record the search tree result.
        # If we need to save memory, we can just save the `search_root.array_index_to_move_index` and 
        # `search_root.children_visit_counts` arrays, and compute the full policy (of length NUM_MOVES)
        # when writing to disk.
        policy = np.zeros((NUM_MOVES,))
        policy[search_root.array_index_to_move_index] = (
            search_root.children_visit_counts / np.sum(search_root.children_visit_counts)
        )
        self.data_recorder.record_rollout_result(
            self.recorder_game_id,
            state,
            policy,
        )

        # Select the move to play now.
        return search_root.get_move_index_to_play(state)
    

class MCTSValuesNode:
    def __init__(self):
        # These values are all populated when get_value_and_expand_children is called.
        self.num_valid_moves = None

        # Shape (NUM_VALID_MOVES,)
        # Usage: self.children_value_sums[self.move_index_to_array_index[move_index]] -> value sum for move_index
        self.move_index_to_array_index: Dict[int, int] = None

        # Shape (NUM_VALID_MOVES,)
        # Usage:
        #   array_index = np.argmax(self.children_value_sums)
        #   self.array_index_to_move_index[array_index] -> move_index associated with the array index
        self.array_index_to_move_index = None
    
        self.children_value_sums = None         # Shape (4, NUM_VALID_MOVES)
        self.children_visit_counts = None       # Shape (NUM_VALID_MOVES,)
        self.children_priors = None             # Shape (NUM_VALID_MOVES,)

        # This populates over time only as nodes are visited.
        self.move_index_to_child_node: Dict[int, MCTSValuesNode] = {}

    def select_child_by_ucb(self, state: State):
        # Exploitation scores are between 0 and 1. 0 means the player has lost every game from this move,
        # 1 means the player has won every game from this move.
        exploitation_scores = np.divide(
            self.children_value_sums[state.player],
            self.children_visit_counts,
            where=(self.children_visit_counts != 0)
        )
        # Where there's no visit count, use a default value. This value isn't 0 because that's "not fair",
        # given random play there's some non-zero expected value so let's use that instead.
        exploitation_scores[self.children_visit_counts == 0] = UCB_DEFAULT_CHILD_VALUE

        # Followed this: https://aaowens.github.io/blog_mcts.html
        sqrt_total_visit_count = np.sqrt(np.sum(self.children_visit_counts) + 1)

        # Children priors are normalized to add up to 1.
        exploration_scores = (
            UCB_EXPLORATION * self.children_priors * sqrt_total_visit_count / (1 + self.children_visit_counts)
        )
        ucb_scores = exploitation_scores + exploration_scores

        move_index_selected = self.array_index_to_move_index[np.argmax(ucb_scores)]
        return move_index_selected

    async def get_value_and_expand_children(self, state: State, inference_interface: InferenceInterface):
        """
        Populate the children arrays by calling the NN, and return
        the value of the current state.
        """
        valid_moves = state.valid_moves_array()

        # Set up the mapping between move indices and array indices.
        self.array_index_to_move_index = np.flatnonzero(valid_moves)
        self.move_index_to_array_index = {
            move_index: array_index
            for array_index, move_index in enumerate(self.array_index_to_move_index)
        }
        self.num_valid_moves = len(self.array_index_to_move_index)

        player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)

        player_pov_values, player_pov_children_prior_logits = await inference_interface.evaluate(player_pov_occupancies)

        # Rotate the player POV values and policy back to the original player's perspective.
        universal_values = player_pov_helpers.values_to_player_pov(player_pov_values, -state.player)
        universal_children_prior_logits = player_pov_helpers.moves_array_to_player_pov(player_pov_children_prior_logits, -state.player)

        # Exclude invalid moves and take the softmax.
        universal_children_priors = softmax(universal_children_prior_logits[valid_moves])

        self.children_value_sums = np.zeros((4, self.num_valid_moves), dtype=float)
        self.children_visit_counts = np.zeros(self.num_valid_moves, dtype=int)
        self.children_priors = universal_children_priors
        return universal_values

    def get_move_index_to_play(self, state: State):
        probabilities = softmax(self.children_visit_counts)
        probabilities /= np.sum(probabilities)

        return self.array_index_to_move_index[np.random.choice(len(probabilities), p=probabilities)]
    
    def add_noise(self):
        self.children_priors = (1 - ROOT_EXPLORATION_FRACTION) * self.children_priors + \
            ROOT_EXPLORATION_FRACTION * np.random.gamma(ROOT_DIRICHLET_ALPHA, 1, NUM_MOVES)