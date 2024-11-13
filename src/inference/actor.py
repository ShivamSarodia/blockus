import numpy as np
import ray
from typing import Tuple
import torch
import time
from typing import Dict

from configuration import config
from neural_net import NeuralNet
from event_logger import log_event

BOARD_SIZE = config()["game"]["board_size"]

@ray.remote
class InferenceActor:
    def __init__(self, network_config: Dict) -> None:
        self.network_config = network_config

        self.model = NeuralNet(network_config).to("mps")
        model_path = network_config["model_path"]
        self.model.load_state_dict(torch.load(model_path))
        
        print(f"Loaded model from file: {model_path}")
    
    def evaluate_batch(self, boards) -> Tuple[np.ndarray, np.ndarray]:
        # Include an extra .copy() here so we don't get a scary PyTorch warning about 
        # non-writeable tensors.
        start_evaluation = time.perf_counter()

        boards_tensor = torch.from_numpy(boards.copy()).to(dtype=torch.float, device="mps")
        with torch.inference_mode():
            values_logits_tensor, policy_logits_tensor = self.model(boards_tensor)
        
        values = torch.softmax(values_logits_tensor, dim=1).cpu().numpy()
        policy_logits = policy_logits_tensor.cpu().numpy()

        log_event("gpu_evaluation", {
            "duration": time.perf_counter() - start_evaluation,
            "batch_size": boards.shape[0],
        })

        return values, policy_logits
