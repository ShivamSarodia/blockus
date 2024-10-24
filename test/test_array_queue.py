import sys
sys.path = ["/Users/shivamsarodia/Dev/blockus/src"] + sys.path

# import os
# os.environ["CONFIG_PATH"] = '/Users/shivamsarodia/Dev/blockus/configs/default.toml'

import time
import random
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
from inference.array_queue import ArrayQueue



def publisher_1(queue):
    queue.init_in_process()

    while True:
        queue.put(1)


def publisher_2(queue):
    queue.init_in_process()

    while True:
        queue.put(2)


def consumer(queue):
    queue.init_in_process()

    while True:
        queue.get()


if __name__ == "__main__":
    queue = ArrayQueue(20, [
        np.empty((), dtype=int),
    ])

    p2 = multiprocessing.Process(target=process2, args=(queue,))
    p2.start()

    p1 = multiprocessing.Process(target=process1, args=(queue,))
    p1.start()

    p1.join()
    p2.join()

    queue.cleanup()
