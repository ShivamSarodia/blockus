import os
import aim
import glob
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import glob
import time

from configuration import config
from training.load_games import load_games_new
from neural_net import NeuralNet

NETWORK_CONFIG = config()["networks"]["default"]
POLICY_LOSS_WEIGHT = config()["training"]["policy_loss_weight"]
BATCH_SIZE = config()["training"]["batch_size"]
LEARNING_RATE = config()["training"]["learning_rate"]

DEVICE = "mps"
TRAIN_TEST_SPLIT = 0.9
NUM_EPOCHS = 5
TEST_MODEL_BATCH_INTERVAL = 2000

def save_model(model, directory, model_name):
    os.makedirs(f"/Users/shivamsarodia/Dev/blockus/data/notebook-models/{directory}", exist_ok=True)
    model_path = f"/Users/shivamsarodia/Dev/blockus/data/notebook-models/{directory}/{model_name}.pt"
    torch.save(model.state_dict(), model_path)

def get_dataset(game_files):
    gamedata = load_games_new(game_files, with_tqdm=True)
    boards_tensor = torch.from_numpy(gamedata["boards"]).to(dtype=torch.float32)
    policies_tensor = torch.from_numpy(gamedata["policies"]).to(dtype=torch.float32)
    values_tensor = torch.from_numpy(gamedata["values"]).to(dtype=torch.float32)
    valid_moves = torch.from_numpy(gamedata["valid_moves"]).to(dtype=torch.bool)
    return torch.utils.data.TensorDataset(boards_tensor, policies_tensor, values_tensor, valid_moves)

def get_test_dataset_no_train(train_dataset, test_dataset):
    train_boards = set()
    for board, _, _, _ in train_dataset:
        train_boards.add(board.numpy(force=True).tobytes())

    test_dataset_indices_not_in_train = []
    for i, (board, _, _, _) in enumerate(test_dataset):
        if board.numpy(force=True).tobytes() not in train_boards:
            test_dataset_indices_not_in_train.append(i)

    return torch.utils.data.Subset(test_dataset, test_dataset_indices_not_in_train)

def get_test_losses(model, test_dataset):
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model.eval()

    results = {
        "total_loss": 0,
        "value_loss": 0,
        "policy_loss": 0,
        "value_max_correct": 0,
        "policy_max_correct": 0,
    }

    total_sample_count = 0

    with torch.inference_mode():
        for boards, policies, values, valid_moves in dataloader:
            boards = boards.to(dtype=torch.float32, device=DEVICE)
            policies = policies.to(dtype=torch.float32, device=DEVICE)
            values = values.to(dtype=torch.float32, device=DEVICE)

            pred_values, pred_policy_logits = model(boards.to(dtype=torch.float32, device=DEVICE))
            value_loss = nn.CrossEntropyLoss(reduction="sum")(
                pred_values,
                values,
            )
            policy_loss = nn.CrossEntropyLoss(reduction="sum")(
                pred_policy_logits,
                policies,
            )
            loss = value_loss + POLICY_LOSS_WEIGHT * policy_loss

            results["total_loss"] += loss.item()
            results["value_loss"] += value_loss.item()
            results["policy_loss"] += policy_loss.item()
            results["value_max_correct"] += (pred_values.argmax(dim=1) == values.argmax(dim=1)).sum().item()
            results["policy_max_correct"] += (pred_policy_logits.argmax(dim=1) == policies.argmax(dim=1)).sum().item()

            total_sample_count += len(boards)

    results["total_loss"] /= total_sample_count
    results["value_loss"] /= total_sample_count
    results["policy_loss"] /= total_sample_count
    results["value_max_correct"] /= total_sample_count
    results["policy_max_correct"] /= total_sample_count

    return results

def train(train_dataset, test_dataset, test_dataset_no_train):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    run = aim.Run(repo='/Users/shivamsarodia/Dev/blockus/')
    run["hparams"] = {
        "training": config()["training"],
        "networks": config()["networks"]["default"],
        "misc": {
            "num_epochs": NUM_EPOCHS,
            "test_train_split": TRAIN_TEST_SPLIT,
            "num_train_games": len(train_dataset),
            "num_test_games": len(test_dataset),
        },
    }

    print("Starting training on run:", run.hash)

    model = NeuralNet(NETWORK_CONFIG)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    batch_index = 0
    for epoch in range(NUM_EPOCHS):
        print("Starting epoch", epoch)
        for boards, policies, values, valid_moves in tqdm(train_dataloader):
            model.train()

            boards = boards.to(dtype=torch.float32, device=DEVICE)
            policies = policies.to(dtype=torch.float32, device=DEVICE)
            values = values.to(dtype=torch.float32, device=DEVICE)

            pred_values, pred_policy = model(boards)
            value_loss = nn.CrossEntropyLoss()(
                pred_values,
                values,
            )
            policy_loss = nn.CrossEntropyLoss()(
                pred_policy,
                policies,
            )
            loss = value_loss + POLICY_LOSS_WEIGHT * policy_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            training_result = {
                "total_loss": loss.item(),
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_max_correct": (pred_values.argmax(dim=1) == values.argmax(dim=1)).sum().item() / len(boards),
                "policy_max_correct": (pred_policy.argmax(dim=1) == policies.argmax(dim=1)).sum().item() / len(boards),
            }

            for key, value in training_result.items():
                run.track(
                    value,
                    name=key,
                    step=batch_index,
                    context={"subset": "train"},
                )

            if batch_index % TEST_MODEL_BATCH_INTERVAL == 0:
                test_losses = get_test_losses(model, test_dataset)
                for key, value in test_losses.items():
                    run.track(
                        value,
                        name=key,
                        step=batch_index,
                        context={"subset": "test"},
                    )

                test_losses_no_train = get_test_losses(model, test_dataset_no_train)
                for key, value in test_losses_no_train.items():
                    run.track(
                        value,
                        name=key,
                        step=batch_index,
                        context={"subset": "test_no_train"},
                    )
            batch_index += 1

        print("Finished epoch, saving model...")
        save_model(model, run.hash, f"epoch_{epoch}")
    run.close()

def load_datasets():
    print("Loading games...")

    file_paths = glob.glob("/Users/shivamsarodia/Dev/blockus/data/2024-11-23_00-37-50-doublehandedness/untrained_games_*/*.npz")

    random.seed(20554)
    random.shuffle(file_paths)

    num_train_games = int(len(file_paths) * TRAIN_TEST_SPLIT)
    train_file_paths = file_paths[:num_train_games]
    test_file_paths = file_paths[num_train_games:]

    print("Num train files:", len(train_file_paths))
    print("Num test files:", len(test_file_paths))

    print("Loading train games...")
    train_dataset = get_dataset(train_file_paths)

    print("Loading test games...")
    test_dataset = get_dataset(test_file_paths)

    print("Generating test dataset that excludes train data...")
    train_boards = set()
    for board, _, _, _ in train_dataset:
        train_boards.add(board.numpy(force=True).tobytes())

    test_dataset_indices_not_in_train = []
    for i, (board, _, _, _) in enumerate(test_dataset):
        if board.numpy(force=True).tobytes() not in train_boards:
            test_dataset_indices_not_in_train.append(i)

    test_dataset_no_train = torch.utils.data.Subset(test_dataset, test_dataset_indices_not_in_train)

    print("Num train samples:", len(train_dataset))
    print("Num test samples:", len(test_dataset))
    print("Num test samples that are not in train data:", len(test_dataset_no_train))

    return train_dataset, test_dataset, test_dataset_no_train

def run():
    train(*load_datasets())
