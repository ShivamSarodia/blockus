import random
import time
import torch
import numpy as np
import os
from typing import Dict
import threading
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

        # Buffers for sharing data between input queue and GPU thread.
        self.input_queue_reader_lock = None  # set in self.run()
        self.occupancies = np.empty((input_queue.capacity, *input_queue.items_like[0].shape))
        self.evaluation_ids = []
        self.process_ids = []

        # Buffers for sharing data between GPU thread and output queue.
        self.output_queue_data_ready = None  # set in self.run()
        self.output_queue_writer_lock = None  # set in self.run()

        # Metrics
        self.batch_size_sum = 0
        self.batch_size_count = 0
        self.batch_size_distribution = {k: 0 for k in range(0, 1000, 10)}
        self.time_spent_evaluating = 0
    
    def run(self):
        print("Running evaluation engine on device: ", self.device)

        # This needs to be set here rather than in the constructor because otherwise
        # multiprocessing gets grumpy when forking.
        self.input_queue_reader_lock = threading.Lock()
        self.output_queue_data_ready = threading.Event()
        self.output_queue_writer_lock = threading.Lock()

        thread = threading.Thread(target = self.run_queue_reader, name="evaluation_input_queue_reader", daemon=True)
        thread.start()

        thread = threading.Thread(target = self.run_queue_writer, name="evaluation_output_queue_writer", daemon=True)
        thread.start()        

        self.model.to(self.device)
        self.model.eval()

        self.input_queue.init_in_process()
        for output_queue in self.output_queue_map.values():
            output_queue.init_in_process()

        if PROFILER_DIRECTORY:
            profiler = pyinstrument.Profiler()
            profiler.start()

        try:
            with torch.inference_mode():
                while True:
                    self.input_queue_reader_lock.acquire(timeout=1)
                    batch_size = len(self.evaluation_ids)

                    # If there's no occupancies ready to evaluate, then immediately 
                    # release the lock and sleep for a bit before trying again.
                    if batch_size == 0:
                        self.input_queue_reader_lock.release()
                        time.sleep(random.random() * 10e-3)
                        continue

                    # (At this point, we still have the input_queue_reader_lock.)

                    # Report some metrics.
                    self.batch_size_sum += batch_size
                    self.batch_size_count += 1
                    self.batch_size_distribution[(batch_size // 10) * 10] += 1

                    # Send the occupancies tensor into the GPU, which also copies it.
                    occupancies_tensor_from_numpy = torch.from_numpy(self.occupancies[:batch_size])
                    occupancies_tensor = occupancies_tensor_from_numpy.to(dtype=torch.float, device=self.device)

                    # It's very important that we made a *copy* of the occupancies tensor rather than 
                    # simply a the same tensor, because the array will be overwritten by the input queue reader thread
                    # as soon as we release the lock.
                    # 
                    # The Torch `to` method sometimes returns `self` and sometimes returns a copy, depending
                    # on if the dtype and device match the original array. To assure confidence here, we verify
                    # that the occupancies_tensor is a copy.    
                    assert occupancies_tensor is not occupancies_tensor_from_numpy

                    # Similarly, reassign the original evaluation and process ID lists to be empty
                    # so that the reader process doesn't clobber values in our lists.
                    evaluation_ids = self.evaluation_ids
                    process_ids = self.process_ids

                    self.evaluation_ids = []
                    self.process_ids = []

                    # Finally, we're free to release our lock. It's important we do this before we run the NN 
                    # evaluation because the evaluation takes a long time.
                    self.input_queue_reader_lock.release()

                    start_time = time.time()
                    values_logits_tensor, policy_logits_tensor = self.model(occupancies_tensor)

                    values = torch.softmax(values_logits_tensor, dim=1).cpu().numpy()
                    policy_logits = policy_logits_tensor.cpu().numpy()

                    self.time_spent_evaluating += (time.time() - start_time)

                    self.output_queue_writer_lock.acquire(timeout=1)
                    self.values_to_serve = values 
                    self.policy_logits_to_serve = policy_logits
                    self.evaluation_ids_to_serve = evaluation_ids
                    self.process_ids_to_serve = process_ids
                    self.output_queue_writer_lock.release()
                    
                    self.output_queue_data_ready.set()

        finally:
            print(f"Average batch size on device {self.device}: {self.batch_size_sum / self.batch_size_count}")
            print(f"Batch size distribution: {self.batch_size_distribution}")
            print(f"Time spent evaluating:  {self.time_spent_evaluating}")
            print(f"Time per evaluation: {self.time_spent_evaluating / self.batch_size_sum}")
            if PROFILER_DIRECTORY:
                profiler.stop()
                path = f"{PROFILER_DIRECTORY}/{int(time.time() * 1000)}_{random.getrandbits(30)}_evaluation_engine_{self.device}.html"
                print(f"Writing profiler info to path: {path}")
                profiler.write_html(path)

    def run_queue_reader(self):
        """
        Reads from the queue and writes into an in-memory buffer for the GPU thread to consume
        faster.
        """
        print("Starting queue reader thread...")
        while True:
            occupancies, evaluation_id, process_id = self.input_queue.get()
            self.input_queue_reader_lock.acquire(timeout=5)
            self.occupancies[len(self.evaluation_ids)] = occupancies
            self.evaluation_ids.append(int(evaluation_id))
            self.process_ids.append(int(process_id))
            self.input_queue_reader_lock.release()

    def run_queue_writer(self):
        print("Starting queue write thread...")
        while True:
            # Wait until there's data ready to serve.
            # The timeout here is fairly large because it blocks on evaluation of 
            # the neural network on a batch, which can take quite a meaningful amount
            # of time to resolve.
            self.output_queue_data_ready.wait(timeout=60)
            self.output_queue_data_ready.clear()
            
            # Acquire the lock to make sure the GPU thread won't try to write fresh
            # output while we're still serving out existing content.
            self.output_queue_writer_lock.acquire(timeout=1)

            # Grab data from where the GPU thread left it and serve it out.
            for value, policy_logit, evaluation_id, process_id in zip(
                self.values_to_serve,
                self.policy_logits_to_serve,
                self.evaluation_ids_to_serve,
                self.process_ids_to_serve,
            ):
                self.output_queue_map[process_id].put(value, policy_logit, evaluation_id)

            # Now that we've served all content, safe to release the write lock again and let the
            # GPU thread write new values.
            self.output_queue_writer_lock.release()
