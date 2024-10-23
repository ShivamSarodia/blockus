import random
import asyncio
import threading
import numpy as np
from typing import Tuple

from inference.array_queue import ArrayQueue

class InferenceInterface:
    def __init__(self, process_id: int, input_queue: ArrayQueue, output_queue: ArrayQueue):
        self.process_id = process_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.task_futures = {}

    async def evaluate(self, player_pov_occupancies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Add the input to the queue, along with the ID of the process that's making
        # the request and an ID that maps this request to the future that will be resolved.
        task_id = random.getrandbits(30)
        future = asyncio.get_event_loop().create_future()
        self.task_futures[task_id] = future

        self.input_queue.put(
            player_pov_occupancies,
            task_id,
            self.process_id,
        )

        result = await future
        del self.task_futures[task_id]

        return result
    
    def init_in_process(self):
        """
        Call this within the process to initialize the input and output queues and
        start the background thread that will resolve the futures.
        """
        self.input_queue.init_in_process()
        self.output_queue.init_in_process()

        thread = threading.Thread(target=self._background_thread, args=[asyncio.get_event_loop()], daemon=True)
        thread.start()

    def _background_thread(self, loop):
        while True:
            value, policy, task_id = self.output_queue.get()
            task_id = int(task_id)
            future = self.task_futures[task_id]
            loop.call_soon_threadsafe(future.set_result, (value, policy))