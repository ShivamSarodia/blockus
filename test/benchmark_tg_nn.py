import os 
import sys

os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/self_play.yaml"
os.environ["CONFIG_OVERRIDES"] = 'game.moves_directory="/Users/shivamsarodia/Dev/blockus/data/moves_10"'
sys.path.append("/Users/shivamsarodia/Dev/blockus/src")

import argparse
import time
import numpy as np
from tinygrad import Tensor
from tinygrad import dtypes
from tinygrad import TinyJit
from typing import Dict 

from configuration import config
from neural_net_tg import NeuralNet

BOARD_SIZE = config()["game"]["board_size"]

parser = argparse.ArgumentParser(description="Benchmark neural network")
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--dtype', type=str, default='float32', help='Data type for tensors')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_BATCHES_WARM_UP = 10
NUM_BATCHES_TO_EVALUATE = 20
DTYPE = getattr(dtypes, args.dtype)

from tinygrad import Device
print(Device.DEFAULT)

model = NeuralNet(config()["networks"]["default"])
# model.set_dtype(DTYPE)
# model.eval()

compiled_model = TinyJit(model)
# compiled_model = model

def time_per_eval(num_batches, batch_size, dtype, compiled_model):
    random_arrays = np.random.random((num_batches, batch_size, 4, BOARD_SIZE, BOARD_SIZE))

    start = time.perf_counter()
    for i in range(num_batches):
        boards = Tensor(random_arrays[i], dtype=DTYPE)
        result = compiled_model(boards)
        if isinstance(result, tuple):
            for x in result:
                x.numpy()
        else:
            result.numpy()

    elapsed = time.perf_counter() - start

    return elapsed / (num_batches * batch_size)

print("Warming up...")
time_per_eval(NUM_BATCHES_WARM_UP, BATCH_SIZE, DTYPE, compiled_model)
print("Evaluating...")
print(time_per_eval(NUM_BATCHES_TO_EVALUATE, BATCH_SIZE, DTYPE, compiled_model))
