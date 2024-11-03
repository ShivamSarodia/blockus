from typing import Tuple
import random
import numpy as np
import asyncio

from configuration import config
from inference.actor import InferenceActor

INFERENCE_CLIENT_BATCH_SIZE = config()["architecture"]["inference_batch_size"]
BOARD_SIZE = config()["game"]["board_size"]

class InferenceClient:
    def __init__(self, actor: InferenceActor):
        self.actor = actor
        
        # This fills up as evaluation requests come in. When full, we submit to
        # an external process. We don't send these one-by-one because each remote()
        # request has some overhead.
        self.evaluation_batch = []
        self.evaluation_batch_ids = []
        self.futures = {}

        # The event loop is set by init_in_process.
        self.loop = None

    async def evaluate(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        evaluation_id = random.getrandbits(60)
        future = self.loop.create_future()
        self.futures[evaluation_id] = future

        self.evaluation_batch.append(board)
        self.evaluation_batch_ids.append(evaluation_id)

        if len(self.evaluation_batch) == INFERENCE_CLIENT_BATCH_SIZE:
            await self._evaluate_batch()

        result = await future
        del self.futures[evaluation_id]

        return result

    def init_in_process(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def _evaluate_batch(self):
        # Make local copies of all the relevant values, and then reset 
        # the self.* copies. This way, when we switch to a different async
        # thread upon calling the await below, we don't need to worry about 
        # our values here getting clobbered.
        evaluation_batch_np = np.stack(self.evaluation_batch)

        # I believe the [:] isn't necessary here, but I'm including it just
        # to be safe.
        evaluation_batch_ids = self.evaluation_batch_ids[:]

        self.evaluation_batch = []
        self.evaluation_batch_ids = []

        values, policy_logits = await self.actor.evaluate_batch.remote(evaluation_batch_np)

        assert len(values) == len(evaluation_batch_ids)

        for value, policy_logit, evaluation_id in zip(
            values,
            policy_logits,
            evaluation_batch_ids,
        ):
            future = self.futures[evaluation_id]
            future.set_result((value, policy_logit))
