import numpy as np
import cProfile
import random
from typing import Dict, NamedTuple
from tqdm import tqdm
import time

import player_pov_helpers
import neural_net
from constants import MOVES, BOARD_SIZE, NUM_MOVES
from state import State
from display import Display
from data_recorder import DataRecorder
from neural_net import NeuralNet
from multiprocessing import Pool


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Config(NamedTuple):
    num_mcts_rollouts: int
    ucb_exploration: float
    ucb_default_child_value: float
    root_dirichlet_alpha: float
    root_exploration_fraction: float


class MCTSAgent:
    def __init__(self, config: Config, model: NeuralNet, data_recorder: DataRecorder):
        self.config = config
        self.model = model
        self.data_recorder = data_recorder

    def select_move_index(self, state: State):
        search_root = MCTSValuesNode(self.config)
        search_root.get_value_and_expand_children(state, self.model)

        for _ in range(self.config.num_mcts_rollouts):
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
                new_node = MCTSValuesNode(self.config)
                value = new_node.get_value_and_expand_children(scratch_state, self.model)
                nodes_visited[-1].move_index_to_child_node[moves_played[-1]] = new_node

            # Now, backpropagate the value up the visited notes.
            for i in range(len(nodes_visited)):
                nodes_visited[i].children_value_sums[:,moves_played[i]] += value 
                nodes_visited[i].children_visit_counts[moves_played[i]] += 1

        # Record the search tree result.
        self.data_recorder.record_rollout_result(
            state,
            search_root.children_visit_counts / np.sum(search_root.children_visit_counts),
        )

        # Select the move to play now.
        return search_root.get_move_index_to_play(state)
    

class MCTSValuesNode:
    def __init__(self, config: Config):
        self.config = config

        # All of these below can be rewritten in terms of num *valid* moves to save a lot
        # of memory if needed.
        #
        # They are populated when get_value_and_expand_children is called.
        self.children_value_sums = None      # Shape (4, NUM_MOVES)
        self.children_visit_counts = None    # Shape (NUM_MOVES,)
        self.children_priors = None          # Shape (NUM_MOVES,)

        # This populates over time only as nodes are visited.
        self.move_index_to_child_node: Dict[int, MCTSValuesNode] = {}

    def select_child_by_ucb(self, state: State):
        # Exploitation scores are between 0 and 1. 0 means the player has lost every game from this move,
        # 1 means the player has won every game from this move.
        exploitation_scores = np.divide(
            self.children_value_sums[state.player],
            self.children_visit_counts,
            # Where there's no visit count, use a default value. This value isn't 0 because that's "not fair",
            # given random play there's some non-zero expected value so let's use that instead.
            out=np.ones_like(self.children_value_sums[state.player]) * self.config.ucb_default_child_value,
            where=(self.children_visit_counts != 0)
        )

        # Followed this: https://aaowens.github.io/blog_mcts.html
        sqrt_total_visit_count = np.sqrt(np.sum(self.children_visit_counts) + 1)

        # Children priors are normalized to add up to 1.
        exploration_scores = (
            self.config.ucb_exploration * self.children_priors * sqrt_total_visit_count / (1 + self.children_visit_counts)
        )
        ucb_scores = exploitation_scores + exploration_scores
        ucb_scores[~state.valid_moves_array()] = -np.inf

        move_index_selected = np.argmax(ucb_scores)
        return move_index_selected

    def get_value_and_expand_children(self, state: State, model: NeuralNet):
        """
        Populate the children arrays by calling the NN, and return
        the value of the current state.
        """
        # TODO: This is supposed to be input to the NN.
        player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(state.occupancies, state.player)

        player_pov_values, player_pov_children_prior_logits = neural_net.evaluate(model, player_pov_occupancies, "mps")

        # Rotate the player POV values and policy back to the original player's perspective.
        universal_values = player_pov_helpers.values_to_player_pov(player_pov_values, -state.player)
        universal_children_prior_logits = player_pov_helpers.moves_array_to_player_pov(player_pov_children_prior_logits, -state.player)

        # Softmax the children priors while excluding invalid moves.
        universal_children_priors = np.zeros((NUM_MOVES,), dtype=float)
        valid_moves = state.valid_moves_array()
        universal_children_priors[valid_moves] = softmax(universal_children_prior_logits[valid_moves])

        self.children_value_sums = np.zeros((4, NUM_MOVES), dtype=float)
        self.children_visit_counts = np.zeros(NUM_MOVES, dtype=int)
        self.children_priors = universal_children_priors
        return universal_values

    def get_move_index_to_play(self, state: State):
        probabilities = softmax(self.children_visit_counts)
        probabilities *= state.valid_moves_array()
        probabilities /= np.sum(probabilities)

        return np.random.choice(len(probabilities), p=probabilities)
    
    def add_noise(self):
        self.children_priors = (1 - self.config.root_exploration_fraction) * self.children_priors + \
            self.config.root_exploration_fraction * np.random.gamma(self.config.root_dirichlet_alpha, 1, NUM_MOVES)


class RandomAgent:
    def select_move_index(self, state: State):
        return state.select_random_valid_move_index()  


def play_game(model: NeuralNet, data_recorder: DataRecorder):
    config = Config(
        num_mcts_rollouts=500,
        # Reasonable guess at an exploration parameter I guess?
        ucb_exploration=1.4,
        # This is 0.25, because we assume that for an unexpanded child there's a 1/4 
        # chance of winning from it.
        ucb_default_child_value=0.25,
        root_dirichlet_alpha=0.03,
        root_exploration_fraction=0.25
    )

    agents = [
        MCTSAgent(config, model, data_recorder),
        MCTSAgent(config, model, data_recorder),
        MCTSAgent(config, model, data_recorder),
        MCTSAgent(config, model, data_recorder),
    ]

    game_over = False
    state = State()
    while not game_over:
        agent = agents[state.player]
        move_index = agent.select_move_index(state)
        game_over = state.play_move(move_index)

    data_recorder.record_game_end(state.result())

PROCESSES = 2

def run(output_data_dir):
    data_recorder = DataRecorder(output_data_dir)
    model = NeuralNet().to("mps")
    try:
        while True:
            print("Playing game...")
            start = time.time()
            play_game(model, data_recorder)
            print(f"Game finished in {time.time() - start:.2f}s")
    except:
        data_recorder.flush()
        raise