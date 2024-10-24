import random
import asyncio
import threading
import os
import time
import numpy as np
from typing import Tuple

from inference.array_queue import ArrayQueue

class InferenceInterface:
    def __init__(self, process_id: int, input_queue: ArrayQueue, output_queue: ArrayQueue):
        self.process_id = process_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.futures = {}

        # The event loop is set by init_in_process.
        self.loop = None

    async def evaluate(self, player_pov_occupancies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Add the input to the queue, along with the ID of the process that's making
        # the request and an ID that maps this request to the future that will be resolved.
        evaluation_id = random.getrandbits(30)
        future = self.loop.create_future()
        self.futures[evaluation_id] = future

        self.input_queue.put(
            player_pov_occupancies,
            evaluation_id,
            self.process_id,
        )

        result = await future
        del self.futures[evaluation_id]

        return result
    
    def init_in_process(self, loop: asyncio.AbstractEventLoop):
        """
        Call this within the process to initialize the input and output queues and
        start the background thread that will resolve the futures.
        """
        self.loop = loop
        self.input_queue.init_in_process()
        self.output_queue.init_in_process()

        thread = threading.Thread(target=self._background_thread, daemon=True)
        thread.start()

    def _background_thread(self):
        while True:
            value, policy, evaluation_id = self.output_queue.get()
            # print(f"{time.time(), os.getpid()}: Got from queue", flush=True)
            evaluation_id = int(evaluation_id)
            future = self.futures[evaluation_id]
            self.loop.call_soon_threadsafe(future.set_result, (value, policy))