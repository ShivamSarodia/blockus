import random
import time
import torch
from typing import Dict

from configuration import config
from inference.array_queue import ArrayQueue
from neural_net import NeuralNet

MAXIMUM_BATCH_SIZE = config()["inference"]["maximum_batch_size"]

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
    
    def run(self):
        print("Running evaluation engine on device: ", self.device)

        self.model.to(self.device)
        self.model.eval()

        self.input_queue.init_in_process()
        for output_queue in self.output_queue_map.values():
            output_queue.init_in_process()

        while True:
            # Get a batch of tasks.
            player_pov_occupancy_batch, task_ids, process_ids = self.input_queue.get_many_nowait(MAXIMUM_BATCH_SIZE)

            if len(player_pov_occupancy_batch) == 0:
                # Sleep for up to 10ms before trying again.
                print("Nothing to evaluate.")
                time.sleep(random.random() * 10e-3)
                continue

            print(f"Evaluation batch size: {len(player_pov_occupancy_batch)}")
            
            # Evaluate the batch to get values and policies.
            values, policies = self._evaluate_batch(player_pov_occupancy_batch)
            
            # Put the results in the corresponding output queue.
            for value, policy, task_id, process_id in zip(values, policies, task_ids, process_ids):
                self.output_queue_map[int(process_id)].put(value, policy, int(task_id))

    def _evaluate_batch(self, player_pov_occupancy_batch):
        occupancies_tensor = torch.from_numpy(player_pov_occupancy_batch).to(dtype=torch.float, device=self.device)
        with torch.inference_mode():
            raw_values, raw_policies = self.model(occupancies_tensor)
        return (
            torch.softmax(raw_values, dim=1).cpu().numpy(),
            raw_policies.cpu().numpy(),
        )
