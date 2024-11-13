import ray
import time
import subprocess
import ray.exceptions
import os
import json

from configuration import config
from inference.client import InferenceClient
from inference.actor import InferenceActor
from gameplay_actor import GameplayActor
from datetime import datetime, timedelta


RUNTIME = config()["development"]["runtime"]
BASE_OUTPUT_DIRECTORY = config()["development"]["output_directory"]
DISPLAY_LOGS_IN_CONSOLE = config()["development"]["display_logs_in_console"]
BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
GAMEPLAY_PROCESSES = config()["architecture"]["gameplay_processes"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]
NETWORKS = config()["networks"]


def run():
    output_data_dir = generate_output_data_dir()

    # To start, save the config to the output data directory.
    with open(output_data_dir + "/config.json", "w") as config_file:
        json.dump(
            config(),
            config_file,
            indent=4,
            separators=(',', ': ')
        )
    print("Saved config file to output directory.")

    ray.init(log_to_driver=DISPLAY_LOGS_IN_CONSOLE)

    # Start the Ray actor that runs GPU computations.
    inference_clients = {}
    for network_name, network_config in NETWORKS.items():
        print(f"Starting inference actor for network '{network_name}'...")
        inference_actor = InferenceActor.remote(network_config)
        inference_clients[network_name] = InferenceClient(inference_actor, network_config["batch_size"])

    # Now, start Ray actors that run gameplay.
    gameplay_actors = [
        GameplayActor.remote(inference_clients, output_data_dir)
        for _ in range(GAMEPLAY_PROCESSES)
    ]

    # Blocks indefinitely, because gameplay actor never finishes.
    try:
        finish_time = (datetime.now() + timedelta(seconds=RUNTIME)).strftime("%I:%M:%S %p")
        print(f"Starting gameplay actors, running for {RUNTIME} seconds, finishing at {finish_time}...")
        ray.get(
            [gameplay_actor.run.remote() for gameplay_actor in gameplay_actors],
            timeout=RUNTIME,
        )
    except (KeyboardInterrupt, ray.exceptions.GetTimeoutError):
        print("Shutting down...")
        print("Cleaning up gameplay actors...")
        ray.get([
            gameplay_actor.cleanup.remote() for gameplay_actor in gameplay_actors
        ])
        print("Done cleaning up gameplay actors.")
        print("Shutting down Ray...")
        ray.shutdown()
        time.sleep(1)
        print("Done shutting down Ray.")

        copy_ray_logs(output_data_dir)

        print("Exiting.")

def generate_output_data_dir():
    random_word = subprocess.run(
        "sort -R /usr/share/dict/words | head -n 1 | tr '[:upper:]' '[:lower:]'",
        shell=True,
        capture_output = True,
        text=True,
    ).stdout.strip()
    output_data_dir = os.path.join(
        BASE_OUTPUT_DIRECTORY,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{random_word}/"
    )
    print(f"\033[94m\033[1mOUTPUT DATA DIRECTORY: {output_data_dir}\033[0m")
    os.makedirs(output_data_dir, exist_ok=True)
    return output_data_dir

def copy_ray_logs(output_data_dir):
    output_file_path = os.path.join(output_data_dir, "logs.txt")
    print(f"Copying Ray logs...")
    with open(output_file_path, "w") as output_file:
        subprocess.run(
            "cat /tmp/ray/session_latest/logs/worker*.out",
            shell=True,
            stdout=output_file,
            stderr=subprocess.PIPE
        )
    print(f"Done copying Ray logs to {output_file_path}.")