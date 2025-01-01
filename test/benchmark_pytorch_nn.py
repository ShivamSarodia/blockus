import os 
import sys

os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/self_play_20.yaml"
# os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/training_unlooped/fewer_value_head_channels_and_flat_layer.yaml"
# os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/training_unlooped/fewer_value_head_channels.yaml"
# os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/training_unlooped/fewer_value_head_flat_layer.yaml"

os.environ["CONFIG_OVERRIDES"] = 'game.moves_directory="/Users/shivamsarodia/Dev/blockus/data/moves_20"'
sys.path.append("/Users/shivamsarodia/Dev/blockus/src")

import argparse
import time
import torch
import numpy as np
from torch import nn
from typing import Dict 

from configuration import config
from neural_net import NeuralNet

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
BATCH_SIZE = config()["training"]["batch_size"]

parser = argparse.ArgumentParser(description="Benchmark PyTorch neural network")
parser.add_argument('--dtype', type=str, default='int8', help='Data type for tensors')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size')

args = parser.parse_args()

NUM_BATCHES_WARM_UP = 10
NUM_BATCHES_TO_EVALUATE = 100
DTYPE = getattr(torch, args.dtype)
BATCH_SIZE = args.batch_size
model = NeuralNet(config()["networks"]["default_1"])
model.to(dtype=DTYPE, device="mps")

def time_per_eval(num_batches, batch_size, dtype, model):
    model.eval()

    random_arrays = np.random.random((num_batches, batch_size, 4, BOARD_SIZE, BOARD_SIZE))

    start = time.perf_counter()
    for i in range(num_batches):
        boards = torch.from_numpy(random_arrays[i]).to(device="mps", dtype=dtype)

        result = model(boards)
        if isinstance(result, tuple):
            for x in result:
                x.numpy(force=True)
        else:
            result.numpy(force=True)

    elapsed = time.perf_counter() - start

    return elapsed / (num_batches * batch_size)

print("Warming up...")
time_per_eval(NUM_BATCHES_WARM_UP, BATCH_SIZE, DTYPE, model)
print("Evaluating...")
print(time_per_eval(NUM_BATCHES_TO_EVALUATE, BATCH_SIZE, DTYPE, model))
