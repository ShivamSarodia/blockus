import os 
import sys

os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/self_play.yaml"
os.environ["CONFIG_OVERRIDES"] = 'game.moves_directory="/Users/shivamsarodia/Dev/blockus/data/moves_10"'
sys.path.append("/Users/shivamsarodia/Dev/blockus/src")

import argparse
import time
import torch
import numpy as np
from torch import nn
from typing import Dict 
# import mlx.core as mx

from configuration import config
from neural_net import NeuralNet

BOARD_SIZE = config()["game"]["board_size"]

parser = argparse.ArgumentParser(description="Benchmark PyTorch neural network")
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--dtype', type=str, default='float32', help='Data type for tensors')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_BATCHES_WARM_UP = 10
NUM_BATCHES_TO_EVALUATE = 500
DTYPE = getattr(torch, args.dtype)

model = NeuralNet(config()["networks"]["default"])
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
