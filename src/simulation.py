import ray
import time
import subprocess
import ray.exceptions
import traceback
import os
import glob
import json

from configuration import config
from inference.client import InferenceClient
from inference.actor import InferenceActor
from training.actor import TrainingActor
from gameplay_actor import GameplayActor
from datetime import datetime, timedelta


RUNTIME = config()["development"]["runtime"]
LOG_SAVE_INTERVAL = config()["logging"]["save_interval"]
BASE_OUTPUT_DIRECTORY = config()["development"]["output_directory"]
DISPLAY_LOGS_IN_CONSOLE = config()["development"]["display_logs_in_console"]
BOARD_SIZE = config()["game"]["board_size"]
NUM_MOVES = config()["game"]["num_moves"]
GAMEPLAY_PROCESSES = config()["architecture"]["gameplay_processes"]
COROUTINES_PER_PROCESS = config()["architecture"]["coroutines_per_process"]
NETWORKS = config()["networks"]
TRAINING_RUN = config()["training"]["run"]


def run():
    output_data_dir = generate_output_data_dir()

    # To start, save the config to the output data directory.
    with open(output_data_dir + f"/config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.json", "w") as config_file:
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

    # If we're supposed to be training, start the Ray actor that trains the network.
    if TRAINING_RUN:
        gamedata_path = os.path.join(output_data_dir, "games/")
        print(f"Starting training actor reading from {gamedata_path}...")
        training_actor = TrainingActor.remote(gamedata_path)
        training_actor.run.remote()

    # Now, start Ray actors that run gameplay.
    gameplay_actors = [
        GameplayActor.remote(i, inference_clients, output_data_dir)
        for i in range(GAMEPLAY_PROCESSES)
    ]
    for gameplay_actor in gameplay_actors:
        gameplay_actor.run.remote()

    if RUNTIME > 0:
        finish_time = (datetime.now() + timedelta(seconds=RUNTIME)).strftime("%I:%M:%S %p")
        print(f"Running for {RUNTIME} seconds, finishing at {finish_time}...")
    else:
        print("Running indefinitely...")

    # Finally, run the main loop.
    start_time = time.time()
    logs_last_copied = start_time

    try:
        while True:
            current_time = time.time()

            # If it's time to wrap up, break.
            if RUNTIME > 0 and current_time > start_time + RUNTIME:
                break

            # If it's been a while since the logs were copied, copy them now.
            if current_time > logs_last_copied + LOG_SAVE_INTERVAL:
                copy_ray_logs(output_data_dir)
                logs_last_copied = current_time

            # Sleep about 60 seconds before checking again.
            time.sleep(min(60, LOG_SAVE_INTERVAL))

    except KeyboardInterrupt:
        print("Got KeyboardInterrupt...")

    finally:
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
    print(f"\033[94m\033[1mOUTPUT DATA DIRECTORY: {BASE_OUTPUT_DIRECTORY}\033[0m")
    os.makedirs(BASE_OUTPUT_DIRECTORY, exist_ok=True)
    return BASE_OUTPUT_DIRECTORY

def copy_ray_logs(output_data_dir):
    output_file_name = f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.txt"
    output_file_path = os.path.join(output_data_dir, output_file_name)

    print(f"Copying Ray logs...")
    with open(output_file_path, "w") as output_file:
        subprocess.run(
            "cat /tmp/ray/session_latest/logs/worker*.out",
            shell=True,
            stdout=output_file,
            stderr=subprocess.PIPE
        )
    print(f"Done copying Ray logs to {output_file_path}.")

    # Delete any previous logs
    for file in glob.glob(os.path.join(output_data_dir, "logs_*.txt")):
        if file.endswith(output_file_name):
            continue
        os.remove(file)
