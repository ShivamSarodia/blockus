import random
from typing import Any, Dict, Optional
import numpy as np

from configuration import config
from state import State
import player_pov_helpers
from data_recorder import DataRecorder
from inference.client import InferenceClient
from event_logger import log_event


NUM_MOVES = config()["game"]["num_moves"]
LOG_MCTS_REPORT_FRACTION = config()["logging"]["mcts_report_fraction"]
LOG_UCB_REPORT = config()["logging"]["ucb_report"]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class MCTSAgent:
    def __init__(
            self,
            config: Dict,
            inference_client: InferenceClient,
            data_recorder: Optional[DataRecorder],
            recorder_game_id: Optional[int],
        ):
        self.mcts_config = config
        self.inference_client = inference_client
        self.data_recorder = data_recorder
        self.recorder_game_id = recorder_game_id

    async def select_move_index(self, state: State, shared_data: Any):
        is_full_move = random.random() < self.mcts_config["full_move_probability"]
        num_rollouts = self.mcts_config["full_move_rollouts"] if is_full_move else self.mcts_config["fast_move_rollouts"]

        if shared_data is None:
            # If there's no start node, create one and expand it.
            search_root = MCTSValuesNode(self.mcts_config)
            await search_root.get_value_and_expand_children(state, self.inference_client, state.turn)
        else:
            # Otherwise, reuse the existing node.
            search_root = shared_data
            
        # If this is a full move, expand the node by clearing previous counts and adding noise.
        # We don't add noise for fast moves, and we don't expand fast moves so that we don't overwrite
        # the counts from the prior turn.
        if is_full_move:
            # This method gets called twice when start_node is None and it's a full move, but internally
            # it's a no-op.
            await search_root.get_value_and_expand_children(state, self.inference_client, state.turn)
            search_root.add_noise()

        for _ in range(num_rollouts):
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

                # If the game is over, we can now assign values based on the final state.
                if game_over:
                    break

                next_node = nodes_visited[-1].move_index_to_child_node.get(move_index)

                # If next_node does not exist, we've finished the tree traversal and it's time to
                # create a new node + backpropagate.
                if not next_node:
                    break

                # If next_node exists, but was last expanded at a prior turn, then during full moves
                # we treat this as a node we've never visited. (During fast moves, we reuse the existing
                # tree no matter when the associated node was expanded.)
                if next_node.expanded_at_turn < state.turn and is_full_move:
                    break

                nodes_visited.append(next_node)

            if game_over:
                value = scratch_state.result()
            else:
                new_node = next_node or MCTSValuesNode(self.mcts_config)
                value = await new_node.get_value_and_expand_children(scratch_state, self.inference_client, state.turn)
                nodes_visited[-1].move_index_to_child_node[moves_played[-1]] = new_node

            # Now, backpropagate the value up the visited notes.
            for i in range(len(nodes_visited)):
                node = nodes_visited[i]
                node_array_index = node.move_index_to_array_index[moves_played[i]]
                node.children_value_sums[:,node_array_index] += value
                node.children_visit_counts[node_array_index] += 1

        # Record the search tree result for full moves.
        if is_full_move and self.data_recorder:
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
        move_index = search_root.get_move_index_to_play(state)

        # Set the supplemental data for the next move.
        if self.mcts_config["reuse_tree"]:
            shared_data = search_root.move_index_to_child_node.get(move_index)
        else:
            shared_data = None

        return move_index, shared_data


class MCTSValuesNode:
    def __init__(self, mcts_config):
        self.mcts_config = mcts_config

        # These values are all populated when get_value_and_expand_children is called.
        self.expanded_at_turn = None

        # Number of valid moves from the state associated with this node.
        self.num_valid_moves = None

        # The values at this node itself, as returned by the neural network.
        self.values = None

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
        self.children_priors = None            # Shape (NUM_VALID_MOVES,)

        # This populates over time only as nodes are visited.
        self.move_index_to_child_node: Dict[int, MCTSValuesNode] = {}

    def _exploitation_scores(self, player):
        # Exploitation scores are between 0 and 1. 0 means the player has lost every game from this move,
        # 1 means the player has won every game from this move.
        exploitation_scores = np.divide(
            self.children_value_sums[player],
            self.children_visit_counts,
            where=(self.children_visit_counts != 0)
        )

        # For children that have no visits, we just use the value of this node itself based on what the
        # NN has reported.
        #
        # For example, if this node nearly always loses for player 1, and we're doing rollouts for player 1,
        # we should assume that an unexplored node is probably a loss for player 1 as well.
        #
        # We should probably tune this a bit. For example, as we explore more children of this node, we could
        # weigh their results more heavily here.
        exploitation_scores[self.children_visit_counts == 0] = self.values[player]

        return exploitation_scores

    def _exploration_scores(self):
        # Followed this: https://aaowens.github.io/blog_mcts.html
        sqrt_total_visit_count = np.sqrt(np.sum(self.children_visit_counts) + 1)

        # Children priors are normalized to add up to 1.
        return (
            self.mcts_config["ucb_exploration"] * self.children_priors * sqrt_total_visit_count /
            (1 + self.children_visit_counts)
        )

    def select_child_by_ucb(self, state: State):
        exploitation_scores = self._exploitation_scores(state.player)
        exploration_scores = self._exploration_scores()

        ucb_scores = exploitation_scores + exploration_scores

        array_index_selected = np.argmax(ucb_scores)
        move_index_selected = self.array_index_to_move_index[array_index_selected]

        if LOG_UCB_REPORT:
            log_event(
                "ucb_report",
                {
                    "player": state.player,
                    "board": state.occupancies.tolist(),
                    "children_visit_counts": self.children_visit_counts.tolist(),
                    "children_value_sums": self.children_value_sums.tolist(),
                    "children_priors": self.children_priors.tolist(),
                    "array_index_to_move_index": self.array_index_to_move_index.tolist(),
                    "values": self.values.tolist(),
                    "exploitation_scores": exploitation_scores.tolist(),
                    "exploration_scores": exploration_scores.tolist(),
                    "ucb_scores": ucb_scores.tolist(),
                    "array_index_selected": int(array_index_selected),
                    "move_index_selected": int(move_index_selected),
                }
            )

        return move_index_selected

    async def get_value_and_expand_children(self, state: State, inference_client: InferenceClient, turn: int):
        """
        Populate the children arrays by calling the NN, and return the value of the current state.

        If this method has already been called on this node (tree reuse), we clear out the children
        arrays but we do not call the neural network again.
        """
        # If we've already expanded this node for this turn, skip everything.
        if self.expanded_at_turn == turn:
            return self.values

        self.expanded_at_turn = turn

        # If we're reusing a node from a prior turn, we already have values for most of the node
        # values so we don't need to recompute these.
        if self.num_valid_moves is None:
            valid_moves = state.valid_moves_array()

            # Set up the mapping between move indices and array indices.
            self.array_index_to_move_index = np.flatnonzero(valid_moves)
            self.move_index_to_array_index = {
                move_index: array_index
                for array_index, move_index in enumerate(self.array_index_to_move_index)
            }
            self.num_valid_moves = len(self.array_index_to_move_index)

            # Rotate the occupancies into the player POV.
            player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)

            # Next, we need an array of the valid moves in the rotated POV. Importantly, our array will be in
            # the same order as self.array_index_to_move_index. This means the result of the `evaluate` call
            # will already be in the universal POV, without needing any additional rotation.
            #
            # I'm a bit worried this reduces the efficacy of caching calls, and instead we must be passing in a
            # sorted array of valid modes to ensure that each time the same board appears we're returning the same
            # result. However, I _think_ that for a given board array one can almost always deduce which player's turn it is,
            # and therefore there's exactly one possibility for the valid moves array we pass in. (An exception might be near
            # the end of a game where some players don't have valid moves? I'm not sure.)
            player_pov_valid_move_indices = player_pov_helpers.moves_indices_to_player_pov(self.array_index_to_move_index, state.player)

            player_pov_values, universal_children_prior_logits = await inference_client.evaluate(
                player_pov_occupancies,
                player_pov_valid_move_indices,
                state.turn,
            )

            # Rotate the player POV values back to the universal perspective.
            universal_values = player_pov_helpers.values_to_player_pov(player_pov_values, -state.player)

            # Take the softmax. Note that invalid moves are already excluded within the evaluate call.
            universal_children_priors = softmax(universal_children_prior_logits)

            self.children_priors = universal_children_priors
            self.values = universal_values
        
        self.children_value_sums = np.zeros((4, self.num_valid_moves), dtype=float)
        self.children_visit_counts = np.zeros(self.num_valid_moves, dtype=int)

        return self.values

    def get_move_index_to_play(self, state: State):
        if random.random() < LOG_MCTS_REPORT_FRACTION:
            log_event(
                "mcts_report",
                {
                    "player": state.player,
                    "board": state.occupancies.tolist(),
                    "children_visit_counts": self.children_visit_counts.tolist(),
                    "children_value_sums": self.children_value_sums.tolist(),
                    "children_priors": self.children_priors.tolist(),
                    "array_index_to_move_index": self.array_index_to_move_index.tolist(),
                    "values": self.values.tolist(),
                }
            )

        temperature = self.mcts_config["move_selection_temperature"]
        if temperature == 0:
            array_index = np.argmax(self.children_visit_counts)
        else:
            weighted_visit_counts = np.power(self.children_visit_counts, 1 / temperature)
            probabilities = weighted_visit_counts / np.sum(weighted_visit_counts)
            array_index = np.random.choice(len(probabilities), p=probabilities)
        return self.array_index_to_move_index[array_index]

    def add_noise(self):
        total_dirichlet_alpha = self.mcts_config["total_dirichlet_alpha"]
        noise = np.random.dirichlet([total_dirichlet_alpha / self.num_valid_moves] * self.num_valid_moves)
        root_exploration_fraction = self.mcts_config["root_exploration_fraction"]
        self.children_priors = (
            (1 - root_exploration_fraction) * self.children_priors +
            root_exploration_fraction * noise
        )