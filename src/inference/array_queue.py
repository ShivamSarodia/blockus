from multiprocessing import shared_memory
from typing import List
import numpy as np
import multiprocessing
import queue
import time

class ArrayQueue:
    def __init__(self, capacity: int, items_like: List[np.ndarray], debug=False):
        self.capacity = capacity
        self.items_like = items_like
        self.debug = debug

        self.shms = []
        for item in items_like:
            shm = shared_memory.SharedMemory(create=True, size=item.nbytes * capacity)
            self.shms.append(shm)

        self.arrays = None

        # Stores a list of the indices of the remaining unused indices in the arrays.
        self.unused_indices = multiprocessing.Queue()
        for i in range(capacity):
            self.unused_indices.put_nowait(i)

        # Stores a list of the indices of values that are on the queue.
        # A value should be in this queue ONLY if the corresponding index in the arrays is 
        # filled.
        self.queue_indices = multiprocessing.Queue()

    def init_in_process(self):
        """
        Call this method in the child process to initialize the Numpy array interface for
        accessing shared memory.
        """
        self.arrays = [
            np.ndarray((self.capacity, *item.shape), dtype=item.dtype, buffer=shm.buf)
            for shm, item in zip(self.shms, self.items_like)
        ]

    def put(self, *items):
        try:
            # TODO: For some unknown reason, this is raising.
            index = self.unused_indices.get_nowait()
        except queue.Empty:
            raise ValueError("No more space in the queue.") from None

        for array, item in zip(self.arrays, items):
            array[index] = item

        # We only add the index to the queue after we've written the data to shared memory
        # to ensure nobody starts to read it before we're done.
        self.queue_indices.put_nowait(index)

    def get_many_nowait(self, max_count):
        """
        Get up to max_count items from the queue.
        """
        indices = []
        try:
            for _ in range(max_count):
                index = self.queue_indices.get_nowait()
                indices.append(index)
        except queue.Empty:
            pass
        
        result = [array[indices].copy() for array in self.arrays]

        # We only add the indices back to the unused_indices queue after we've read the 
        # data out of the arrays to ensure nobody else overwrites it first.
        for index in indices:
            self.unused_indices.put_nowait(index)
        
        return result
    
    def get(self):
        """
        Get an item from the queue (in a blocking manner).
        """
        index = self.queue_indices.get()
        result = [np.copy(array[index]) for array in self.arrays]

        # We only add the indices back to the unused_indices queue after we've read the 
        # data out of the arrays to ensure nobody else overwrites it first.
        self.unused_indices.put_nowait(index)
        if self.debug:
            print(f"{time.time()}: Returned index to unused:", index)
        return result
    
    def cleanup(self):
        for shm in self.shms:
            shm.unlink()