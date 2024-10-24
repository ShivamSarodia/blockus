import torch
import torch.multiprocessing
from typing import List
import numpy as np
import queue
import time

class ArrayQueue:
    def __init__(self, capacity: int, items_like: List[np.ndarray]):
        self.capacity = capacity
        self.items_like = items_like

        self.queue = torch.multiprocessing.Queue(capacity)

    def init_in_process(self):
        pass

    def _to_torch(self, items):
        return torch.from_numpy(np.concatenate([np.array(item).flatten() for item in items]))
    
    def _from_torch(self, tensor):
        result = []
        start = 0
        for item in self.items_like:
            end = start + int(np.prod(item.shape))
            result.append(np.copy(tensor[start:end].view(item.shape)))
            start = end
        return result

    def put(self, *items):
        """
        Put an item into the queue.
        """
        self.queue.put(self._to_torch(items))

    def get_many_nowait(self, max_count):
        """
        Get up to max_count items from the queue.
        """
        results = []
        try:
            for _ in range(max_count):
                results.append(
                    self._from_torch(
                        self.queue.get_nowait()
                    )
                )
        except queue.Empty:
            pass
        
        return results
    
    def get(self):
        """
        Get an item from the queue (in a blocking manner).
        """
        return self._from_torch(self.queue.get())

    def cleanup(self):
        pass