import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from neural_net import NeuralNet
import matplotlib.pyplot as plt

# 128 minimizes time per sample on my MacBook's GPU. We may want a different
# configuration when training on a real GPU.
BATCH_SIZE = 128

def _load_paths_as_tensor(directory, paths):
    data = []
    for path in paths:
        data.append(np.load(os.path.join(directory, path)))
    return torch.Tensor(np.concatenate(data))

def _load_datasets(games_dir):
    print("Loading data...")

    directories = [
        "occupancies",
        "children_visit_distributions",
        "values"
    ]
    test_data = []
    train_data = []
    for directory in directories:
        dir_path = os.path.join(games_dir, directory)
        files = sorted(os.listdir(dir_path))

        test_data.append(_load_paths_as_tensor(dir_path, [files[0]]))
        train_data.append(_load_paths_as_tensor(dir_path, files[1:]))
    
    print("Data loaded.")

    return TensorDataset(*train_data), TensorDataset(*test_data)


def _print_losses(value_loss, policy_loss):
    print("   Avg value loss:  ", value_loss)
    print("   Avg policy loss: ", policy_loss)
    print("   Avg total loss:  ", value_loss + policy_loss)


def _train(dataloader, optimizer, model, value_losses, policy_losses):
    size = len(dataloader.dataset)
    model.train()
    for batch, (occupancies, children_visits, values) in enumerate(dataloader):
        start_time = time.time()

        batch_size = len(occupancies)

        occupancies = occupancies.to("mps")
        children_visits = children_visits.to("mps")
        values = values.to("mps")

        pred_values, pred_children_visits = model(occupancies)
        value_loss = nn.CrossEntropyLoss()(pred_values, values)
        policy_loss = nn.CrossEntropyLoss()(pred_children_visits, children_visits)
        loss = value_loss + policy_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current = (batch + 1) * batch_size 
        runtime = time.time() - start_time
        print(f"Training [{current:>5d}/{size:>5d}] [{runtime / batch_size:e}s per sample]")
        _print_losses(
            value_loss.item(),
            policy_loss.item(),
        )
    value_losses.append(value_loss.item())
    policy_losses.append(policy_loss.item())

def _test(dataloader, model, value_losses, policy_losses):
    size = len(dataloader.dataset)
    model.eval()
    sum_value_loss, sum_policy_loss = 0.0, 0.0
    with torch.no_grad():
        for occupancies, children_visits, values in dataloader:
            occupancies = occupancies.to("mps")
            children_visits = children_visits.to("mps")
            values = values.to("mps")

            pred_values, pred_children_visits = model(occupancies)         
            sum_value_loss += nn.CrossEntropyLoss(reduction="sum")(pred_values, values).item()
            sum_policy_loss += nn.CrossEntropyLoss(reduction="sum")(pred_children_visits, children_visits).item()

    value_loss = sum_value_loss / size
    policy_loss = sum_policy_loss / size

    value_losses.append(value_loss)
    policy_losses.append(policy_loss)
    print("Test Error:")
    _print_losses(
        value_loss,
        policy_loss,
    )

def run(games_dir, output_dir):
    train_dataset, test_dataset = _load_datasets(games_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNet().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    test_value_losses = []
    test_policy_losses = []
    train_value_losses = []
    train_policy_losses = []

    try:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            _train(train_dataloader, optimizer, model, train_value_losses, train_policy_losses)
            _test(test_dataloader, model, test_value_losses, test_policy_losses)
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Plotting the value loss, policy loss, winner accuracy, and move accuracy over epochs side by side
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # axs[0, 0].plot(range(1, t + 1), test_value_losses, label='Test value Loss', color='blue')
    # axs[0, 0].set_xlabel('Epoch')
    # axs[0, 0].set_ylabel('Loss')
    # axs[0, 0].set_title('Value Loss over Time')
    # axs[0, 0].legend()
    
    # axs[0, 1].plot(range(1, t + 1), test_policy_losses, label='Test policy Loss', color='orange')
    # axs[0, 1].set_xlabel('Epoch')
    # axs[0, 1].set_ylabel('Loss')
    # axs[0, 1].set_title('Policy Loss over Time')
    # axs[0, 1].legend()
    
    # axs[1, 0].plot(range(1, t + 1), train_value_losses, label='Train value Loss', color='blue')
    # axs[1, 0].set_xlabel('Epoch')
    # axs[1, 0].set_ylabel('Loss')
    # axs[1, 0].set_title('Value Loss over Time')
    # axs[1, 0].legend()
    
    # axs[1, 1].plot(range(1, t + 1), train_policy_losses, label='Train policy Loss', color='orange')
    # axs[1, 1].set_xlabel('Epoch')
    # axs[1, 1].set_ylabel('Loss')
    # axs[1, 1].set_title('Policy Loss over Time')
    # axs[1, 1].legend()
    
    # plt.tight_layout()
    # plt.show()

    os.makedirs(output_dir, exist_ok=True)
    key = str(int(time.time() * 1000))
    torch.save(model.state_dict(), os.path.join(output_dir, f"{key}.pth"))
    
    print("Done!")