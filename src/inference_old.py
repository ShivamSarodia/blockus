import asyncio
import torch
from typing import Tuple
import numpy as np

from config import config 

BOARD_SIZE = config()["game"]["board_size"]
DEBUG_MODE = config()["debug_mode"]

class InferenceEngine:
    def __init__(
            self,
            model,
            device,
            # How many evaluation requests trigger an evaluation.
            batch_size=10,
            # How long to sit on an incomplete batch before triggering an
            # evaluation anyway.
            batch_timeout=1.0,
        ) -> None:
        self.model = model.to(device)
        self.model.eval()

        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # The default event loop.
        self.loop = asyncio.get_event_loop()

        # Queue for incoming evaluation requests that aren't yet moved to a batch.
        self.queue = asyncio.Queue()

    def start_processing(self):
        """
        Start the inference engine. This method will start the event loop
        and begin processing incoming evaluation requests.
        """
        # Python docs seem to indicate we have to save this to a variable so the task is not
        # garbage collected.
        self.consume_queue_task = self.loop.create_task(self._consume_queue())
        self.consume_queue_task.add_done_callback(lambda task: task.result())

    async def evaluate(self, player_pov_occupancies) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a single occupancies array on the neural network
        and returns a (value, policy) pair. It does not block the
        event loop.

        This method is not thread-safe although it is asyncio-task safe (obviously).

        Internally, this method will batch multiple simultaneous invocations
        of this method together to improve performance.
        """
        future = self.loop.create_future()
        # Add to the queue an occupancies array and the future to resolve when the
        # evaluation is complete.
        await self.queue.put((player_pov_occupancies, future))
        return await future
    
    async def _consume_queue(self):
        """
        Consume the queue of incoming evaluation requests. Create batches of the correct
        size and trigger processing in a different thread.
        """
        while True:
            batch = np.zeros((self.batch_size, 4, BOARD_SIZE, BOARD_SIZE), dtype=bool)
            batch_index = 0

            futures = []

            while batch_index < self.batch_size:
                try:
                    player_pov_occupancies, future = await asyncio.wait_for(self.queue.get(), timeout=1)
                    batch[batch_index] = player_pov_occupancies
                    futures.append(future)
                    batch_index += 1
                except asyncio.TimeoutError:
                    if DEBUG_MODE:
                        print("Got a timeout.")
                    break

            if batch_index == 0:
                if DEBUG_MODE:
                    print("Batch is empty.")
                continue

            if DEBUG_MODE and batch_index != self.batch_size:
                print("Batch is not full but running anyway -- suspicious.")

            # We copy the batch to be sure we're not going to have
            # thread safety issues if our main thread modifies the batch before the
            # _evaluate_batch thread runs.
            final_batch = batch[:batch_index].copy()

            batch_index = 0

            # Evaluate the batch.
            values, policies = await asyncio.to_thread(self._evaluate_batch, final_batch)

            # Resolve the futures.
            for future, value, policy in zip(futures, values, policies):
                future.set_result((value, policy))

    def _evaluate_batch(self, batch_occupancies):
        """
        Evaluate a batch of occupancies arrays on the neural network. This method
        runs in a different thread than the rest of the code.
        """
        occupancies_tensor = torch.from_numpy(batch_occupancies).to(dtype=torch.float, device=self.device)
        with torch.inference_mode():
            raw_values, raw_policies = self.model(occupancies_tensor)
        return (
            torch.softmax(raw_values, dim=1).cpu().numpy(),
            raw_policies.cpu().numpy(),
        )
