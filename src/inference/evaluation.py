import random
import time
import torch
import numpy as np
import os
from typing import Dict
import pyinstrument
import pyinstrument.renderers

from configuration import config
from inference.array_queue import ArrayQueue
from neural_net import NeuralNet

MAXIMUM_BATCH_SIZE_ON_GPU = config()["inference"]["maximum_batch_size_on_gpu"]
MAXIMUM_BATCH_SIZE_ON_CPU = config()["inference"]["maximum_batch_size_on_cpu"]
PROFILER_DIRECTORY = config()["development"]["profiler_directory"]

class EvaluationEngine:
    def __init__(self, model: NeuralNet, device: str, input_queue: ArrayQueue, output_queue_map: Dict[int, ArrayQueue]):
        """
        The evaluation engine reads off the input_queue, performs an evaluation, then writes 
        the result to the corresponding output queue.
        
        output_queue_map maps process IDs to output queues
        """
        self.model = model
        self.device = device

        self.input_queue = input_queue
        self.output_queue_map = output_queue_map

        self.maximum_batch_size = MAXIMUM_BATCH_SIZE_ON_CPU if device == "cpu" else MAXIMUM_BATCH_SIZE_ON_GPU
        self.batch_size_sum = 0
        self.batch_size_count = 0
    
    def run(self):
        print("Running evaluation engine on device: ", self.device)

        self.model.to(self.device)
        self.model.eval()

        self.input_queue.init_in_process()
        for output_queue in self.output_queue_map.values():
            output_queue.init_in_process()

        if PROFILER_DIRECTORY:
            profiler = pyinstrument.Profiler()
            profiler.start()

        try:
            while True:
                # Get a batch of tasks.
                results = self.input_queue.get_many_nowait(self.maximum_batch_size)

                batch_size = len(results)
                if batch_size == 0:
                    # Sleep for up to 10ms before trying again.
                    time.sleep(random.random() * 10e-3)
                    continue

                self.batch_size_sum += batch_size
                self.batch_size_count += 1
                
                # Evaluate the batch to get values and policies.
                values, policies = self._evaluate_batch(np.stack([result[0] for result in results]))
                
                # Put the results in the corresponding output queue.
                for value, policy, result in zip(values, policies, results):
                    # k = random.getrandbits(30)
                    # print(f"{time.time(), os.getpid(), int(k)}: Putting in queue start", flush=True)
                    self.output_queue_map[int(result[2])].put(value, policy, int(result[1]))
                    # print(f"{time.time(), os.getpid(), int(k)}: Putting in queue end", flush=True)
        finally:
            print(f"Average batch size on device {self.device}: {self.batch_size_sum / self.batch_size_count}")
            if PROFILER_DIRECTORY:
                profiler.stop()
                profiler.write_html(
                    f"{PROFILER_DIRECTORY}/{int(time.time() * 1000)}_evaluation_engine_{self.device}.html",
                )

    def _evaluate_batch(self, player_pov_occupancy_batch):
        occupancies_tensor = torch.from_numpy(player_pov_occupancy_batch).to(dtype=torch.float, device=self.device)
        with torch.inference_mode():
            raw_values, raw_policies = self.model(occupancies_tensor)

        # Without this, it will look like the cpu() call is slow. In fact, Torch is just streaming the data
        # from the GPU back to the CPU.
        #
        # if PROFILER_DIRECTORY:
        #     torch.mps.synchronize()
        return (
            torch.softmax(raw_values, dim=1).cpu().numpy(),
            raw_policies.cpu().numpy(),
        )
