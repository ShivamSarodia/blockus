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

def process1(queue):
    queue.init_in_process()

    for i in range(10):
        print(f"Writing {i} to shared", flush=True)
        queue.put(
            np.ones((2, 3, 3), dtype=int) * i,
            i * 10,
            i * 100,
        )
        print(f"Wrote {i} to shared", flush=True)

def process2(queue):
    queue.init_in_process()

    for _ in range(10):
        values = queue.get()
        print("Got values: ", values, flush=True)


if __name__ == "__main__":
    queue = ArrayQueue(10, [
        np.empty((2, 3, 3), dtype=int),
        np.empty((), dtype=int),
        np.empty((), dtype=int),
    ])

    p2 = multiprocessing.Process(target=process2, args=(queue,))
    p2.start()

    p1 = multiprocessing.Process(target=process1, args=(queue,))
    p1.start()

    p1.join()
    p2.join()

    queue.cleanup()

# class ThinWrapper:
#     def __init__(self):
#         self.shm = shared_memory.SharedMemory(create=True, size=100)

# def process1(wrapper):
#     n = int(random.getrandbits(8))
#     wrapper.shm.buf[0] = n
#     print("Write to shared", n, flush=True)

# def process2(wrapper):
#     print("Read from shared", wrapper.shm.buf[0], flush=True)

# if __name__ == "__main__":
#     wrapper = ThinWrapper()

#     p1 = multiprocessing.Process(target=process1, args=(wrapper,))
#     p1.start()
#     p1.join()

#     p2 = multiprocessing.Process(target=process2, args=(wrapper,))
#     p2.start()
#     p2.join()

#     wrapper.shm.unlink()

# import multiprocessing

# def child_process_1(shared_memory):
#     # Access and modify the shared memory
#     shared_memory.buf[0] = 7

# def child_process_2(shared_memory):
#     print("Value hi")
#     print(shared_memory.buf[0])

# if __name__ == "__main__":
#     # Create a shared memory block
#     shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=10)

#     # Create a child process
#     process = multiprocessing.Process(target=child_process_1, args=(shared_memory,))
#     process.start()
#     process.join()

#     # Access the shared memory from the parent process
#     process = multiprocessing.Process(target=child_process_2, args=(shared_memory,))
#     process.start()
#     process.join()

#     # Clean up
#     shared_memory.close()
#     shared_memory.unlink()