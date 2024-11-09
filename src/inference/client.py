from typing import Tuple
import time
import random
import logging
import numpy as np
import asyncio
import cacheout.lru

from configuration import config
from inference.actor import InferenceActor
from event_logger import log_event

INFERENCE_CLIENT_BATCH_SIZE = config()["architecture"]["inference_batch_size"]
INFERENCE_RECENT_CACHE_SIZE = config()["architecture"]["inference_recent_cache_size"]
USE_CACHE = INFERENCE_RECENT_CACHE_SIZE > 0
BOARD_SIZE = config()["game"]["board_size"]

class InferenceClient:
    def __init__(self, actor: InferenceActor):
        self.actor = actor
        
        # This fills up as evaluation requests come in. When full, we submit to
        # an external process. We don't send these one-by-one because each remote()
        # request has some overhead.
        self.cache_blocked = 0
        self.evaluation_batch = []
        self.move_indices_batch = []
        self.evaluation_batch_ids = []
        self.futures = {}

        # self.last_evaluation_time = time.perf_counter()

        # These fields are set by init_in_process.
        self.loop = None
        self.evaluation_cache = None

    async def evaluate(self, board: np.ndarray, move_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        board - An occupancies board rolled and rotated into the perspective of the current player.
        move_indices - The move indices (as could be played on the rotated/rolled board) that we're interested
                       in the values for. Usually, this is the list of the valid moves on the rotated/rolled board.
                       This parameter is necessary for caching purposes, so that we only cache the moves that are
                       relevant. This method returns the policy logits in the same order as this list.
        """
        if USE_CACHE:
            cache_key = board.tobytes() + move_indices.tobytes()
            cached_result = self.evaluation_cache.get(cache_key)

        # The cached result is already done, so just return that.
        if USE_CACHE and cached_result and cached_result.done():
            # log_event("evaluate_cache_lookup", {
            #     "cache_key": str(cache_key),
            #     "result": "done",
            #     "type": "done",
            # })
            return cached_result.result()
        
        # We found a value in the cache, but it isn't done yet. That means this result
        # is likely in the currently pending batch of evaluations.
        elif USE_CACHE and cached_result:
            # log_event("evaluate_cache_lookup", {
            #     "cache_key": str(cache_key),
            #     "result": "future",
            #     "type": str(type(cached_result)),
            # })            
            self.cache_blocked += 1
            future = cached_result
            evaluation_id = None
        
        # The value is not in the cache, so we need to add it to the evaluation
        # batch.
        else:
            # log_event("evaluate_cache_lookup", {
            #     "cache_key": str(cache_key),
            #     "result": "none",
            #     "type": "none",
            # })

            evaluation_id = random.getrandbits(60)
            future = self.loop.create_future()
            self.futures[evaluation_id] = future

            if USE_CACHE:
                self.evaluation_cache.set(cache_key, future)

            self.evaluation_batch.append(board)
            self.move_indices_batch.append(move_indices)
            self.evaluation_batch_ids.append(evaluation_id)

        # Check if we're ready to evaluate the batch based on the batch size and the
        # number of evaluations currently waiting on cached values.
        if (
            len(self.evaluation_batch) > 0
        ) and (
            len(self.evaluation_batch) + self.cache_blocked >= INFERENCE_CLIENT_BATCH_SIZE
        ):
            evaluation_params = self._fetch_and_clear_evaluation_params()
            await self._evaluate_batch(*evaluation_params)

        result = await future
        if evaluation_id:
            del self.futures[evaluation_id]

        return result

    def init_in_process(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        if USE_CACHE:
            self.evaluation_cache = cacheout.lru.LRUCache(maxsize=INFERENCE_RECENT_CACHE_SIZE)

    def _fetch_and_clear_evaluation_params(self):
        # Make local copies of all the relevant values, and then reset 
        # the self.* copies. This way, when we switch to a different async
        # thread upon calling the await below, we don't need to worry about 
        # our values here getting clobbered.
        #
        # I believe the [:] isn't necessary below, but I'm including it just
        # to be safe.
        # 
        # This function is not async, to ensure it runs before the event loop might
        # interrupt.
        evaluation_batch_np = np.stack(self.evaluation_batch)
        evaluation_batch_ids = self.evaluation_batch_ids[:]
        move_indices_batch = self.move_indices_batch[:]

        log_event(
            "fetch_evaluation_params",
            {
                "batch_size": len(evaluation_batch_np),
                "cache_blocked": self.cache_blocked,
            },
        )

        self.cache_blocked = 0
        self.evaluation_batch = []
        self.move_indices_batch = []
        self.evaluation_batch_ids = []

        # self.last_evaluation_time = time.perf_counter()

        return evaluation_batch_np, evaluation_batch_ids, move_indices_batch

    async def _evaluate_batch(self, evaluation_batch_np, evaluation_batch_ids, move_indices_batch):
        values, policy_logits = await self.actor.evaluate_batch.remote(evaluation_batch_np)

        assert len(values) == len(evaluation_batch_ids)

        for value, policy_logit, move_indices, evaluation_id in zip(
            values,
            policy_logits,
            move_indices_batch,
            evaluation_batch_ids,
        ):
            filtered_policy_logits = policy_logit[move_indices]
            future = self.futures[evaluation_id]
            future.set_result((value, filtered_policy_logits))
