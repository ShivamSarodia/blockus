import numpy as np
import random
import time
import asyncio
import multiprocessing
import multiprocessing.shared_memory as shared_memory
import multiprocessing.sharedctypes as sharedctypes
from typing import Tuple
from ctypes import c_uint32

from config import config

BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
INPUT_QUEUE_CAPACITY = config()["architecture"]["evaluation_input_queue_capacity"]
OUTPUT_QUEUE_CAPACITY = config()["architecture"]["evaluation_output_queue_capacity"]
MINIMUM_BATCH_SIZE = config()["inference"]["minimum_batch_size"]
MAXIMUM_BATCH_SIZE = config()["inference"]["maximum_batch_size"]

class RingQueue:
    """
    This is a process-safe ring buffer designed to store Numpy arrays and integer arrays.
    """
    def __init__(self, capacity, np_specs, int_array_specs):
        self.capacity = capacity

        # Create buffers for all numpy arrays
        self.np_arrays = []
        self.shared_buffers = []
        for shape, dtype in np_specs:
            size = np.empty((capacity, *shape), dtype=dtype).nbytes
            shm = shared_memory.SharedMemory(create=True, size=size)
            self.shared_buffers.append(shm)
            np_array = np.empty((capacity, *shape), dtype=dtype, buffer=shm.buf)
            self.np_arrays.append(np_array)

        # Create buffers for all integer arrays
        self.int_arrays = [sharedctypes.RawArray(dtype, [0] * capacity) for dtype in int_array_specs]

        # The data is contained in the array from start_index to end_index, excluding end_index.
        # So e.g. if start_index == end_index, the buffer is empty.
        #
        # We don't have these values lock because we're using a lock to protect each method as 
        # a whole.
        self.start_index = sharedctypes.RawValue(c_uint32, 0, lock=False)
        self.end_index = sharedctypes.RawValue(c_uint32, 0, lock=False)
        self.read_write_lock = multiprocessing.RLock()

    def lock(self):
        return self.read_write_lock

    def put(self, *items):
        with self.read_write_lock:
            end_index_value = self.end_index.value
            
            # Store data in each numpy array
            for arr, itm in zip(self.np_arrays, items):
                arr[end_index_value] = itm
            
            # Store data in each integer array
            for arr, val in zip(self.int_arrays, items[len(self.np_arrays):]):
                arr[end_index_value] = val
            
            self.end_index.value = (end_index_value + 1) % self.capacity
        
    def get(self, num_items):
        with self.read_write_lock:
            start = self.start_index.value
            end = (self.start_index.value + num_items) % self.capacity

            if start <= end:
                np_items = [arr[start:end].copy() for arr in self.np_arrays]
                int_values = [arr[start:end] for arr in self.int_arrays]
            else:
                np_items = [np.concatenate((arr[start:], arr[:end])) for arr in self.np_arrays]
                int_values = [arr[start:] + arr[:end] for arr in self.int_arrays]
            
            self.start_index.value = end
            return *np_items, *int_values

    def size(self):
        with self.read_write_lock:
            return (self.end_index.value - self.start_index.value) % self.capacity


# Generate a queue that stores occupancy arrays.
def generate_occupancy_queue():
    return RingQueue(
        INPUT_QUEUE_CAPACITY,
        [
            [(4, BOARD_SIZE, BOARD_SIZE), bool],
        ],
        [
            # Priority ID
            c_uint32,
            # Task ID
            c_uint32,
        ]
    )


# Generate a queue that stores policy results.
def generate_result_queue():
        return RingQueue(
        OUTPUT_QUEUE_CAPACITY,
        [
            [(4,), float],
            [(NUM_MOVES,), float],
        ],
        [
            # Task ID only. 
            # We don't need priority ID here because each result queue is for a 
            # specific PID already.
            c_uint32,
        ]
    )


# We initiate one instance of InferenceEngine in each process that's doing evaluations,
# after we've spawned the process.
class InferenceEngine:
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

        # This dictionary doesn't need to be process-safe because it's only accessed from the 
        # single process using this InferenceEngine.
        # 
        # TODO: Does this need to be thread safe though?
        self.task_futures = {}

    def initialize(self):
        pass

    async def evaluate(self, player_pov_occupancies) -> Tuple[np.ndarray, np.ndarray]:
        # Add the input to the queue, along with the ID of the process that's making
        # the request and an ID that maps this request to the future that will be resolved.
        task_id = random.getrandbits(30)
        future = asyncio.get_event_loop().create_future()

        self.task_futures[task_id] = future

        self.input_queue.put(
            player_pov_occupancies,
            multiprocessing.current_process().pid,
            task_id,            
        )

        result = await future
        del self.task_futures[task_id]

        return result
    

# This engine runs in once in each process responsible for evaluating
# the neural networks.
class EvaluationEngine:
    def __init__(self, input_queue: RingQueue, output_queue_map: RingQueue):
        self.input_queue = input_queue
        self.output_queue_map = output_queue_map

    def run(self):
        while True:
            with self.input_queue.lock():
                size = self.input_queue.size()
                if size < MINIMUM_BATCH_SIZE:
                    continue

                # Get a batch of tasks.
                player_pov_occupancies, process_ids, task_ids = self.input_queue.get(min(MAXIMUM_BATCH_SIZE, size))

                # Evaluate the batch.
                values, policies = self._evaluate_batch(player_pov_occupancies)

                # Put the results in the corresponding output queue.
                for process_id, task_id, value, policy in zip(player_pov_occupancies, process_ids, task_ids, values, policies):
                    self.output_queue_map.put(
                        value,
                        policy,
                        task_id,
                    )
            
            time.sleep(0.01)
            # Check if there's enough tasks in the input queue to form a batch.

            # Evaluate the task.

            # Put the result in the output queue.

    def _evaluate_batch(self):
        raise NotImplementedError()