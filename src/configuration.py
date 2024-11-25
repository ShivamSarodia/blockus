import os
import numpy as np
import yaml
import json
import copy

_CONFIG = None
_MOVES = None

def merge_into_dict(original_dict, new_values):
    assert isinstance(original_dict, dict), f"{original_dict} must be a dict"
    assert isinstance(new_values, dict), f"{new_values} must be a dict"
    for key in new_values:
        if key in original_dict:
            if isinstance(original_dict[key], dict):
                merge_into_dict(original_dict[key], new_values[key])
            else:
                original_dict[key] = new_values[key]
        else:
            original_dict[key] = new_values[key]

def _load_config():
    global _CONFIG
    config = {}

    config_paths = os.environ["CONFIG_PATHS"].split(",")
    config_overrides = os.environ["CONFIG_OVERRIDES"].split(",") if "CONFIG_OVERRIDES" in os.environ else []

    for config_path in config_paths:
        with open(config_path, "rb") as f:
            config_level = yaml.safe_load(f)
            merge_into_dict(config, config_level)

    # Now, apply each override to the config.
    for override in config_overrides:
        key, value = override.split("=")
        keys = key.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        
        assert keys[-1] in current, "Override key not found in config: " + override
        current[keys[-1]] = json.loads(value)

    # Generate a completed list of network configurations from the original config.
    default_network = config["default_network"]
    config["networks"] = {}
    for individual_network_name, individual_network in config["individual_networks"].items():
        network_config = copy.deepcopy(default_network)
        merge_into_dict(network_config, individual_network)
        config["networks"][individual_network_name] = network_config

    del config["default_network"]
    del config["individual_networks"]

    # Finally, generate a completed list of agent configurations from the original config.
    default_agent = config["default_agent"]
    config["agents"] = []
    for individual_agent in config["individual_agents"]:
        agent = copy.deepcopy(default_agent)
        merge_into_dict(agent, individual_agent)
        config["agents"].append(agent)

    del config["default_agent"]
    del config["individual_agents"]

    _CONFIG = config
    print("Loaded config: ", json.dumps(_CONFIG))

def config():
    global _CONFIG

    if _CONFIG is None:
        _load_config()

    return _CONFIG

def moves_data():
    global _MOVES

    if _MOVES is None:
        _load_moves()

    return _MOVES

def _load_moves():
    global _MOVES

    moves_dir = config()["game"]["moves_directory"]
    filenames = [f for f in os.listdir(moves_dir) if os.path.isfile(os.path.join(moves_dir, f))]

    moves = {}
    for filename in filenames:
        if filename.endswith(".npy"):
            key = filename[:-4]
            with open(f"{moves_dir}/{filename}", "rb") as f:
                print("Loading file:", key)
                moves[key] = np.load(f)

    _MOVES = moves

    assert _MOVES["new_occupieds"].shape[0] == config()["game"]["num_moves"]
    assert _MOVES["new_occupieds"].shape[1] == config()["game"]["board_size"]
