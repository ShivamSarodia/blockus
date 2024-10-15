import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from neural_net import NeuralNet
import matplotlib.pyplot as plt

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


def _train(dataloader, optimizer, model):
    size = len(dataloader.dataset)
    model.train()
    for batch, (occupancies, children_visits, values) in enumerate(dataloader):
        pred_values, pred_children_visits = model(occupancies)
        value_loss = nn.CrossEntropyLoss()(pred_values, values)
        policy_loss = nn.CrossEntropyLoss()(pred_children_visits, children_visits)
        loss = value_loss + policy_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(occupancies)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def _test(dataloader, model, value_losses, policy_losses, winner_accuracies, move_accuracies):
    size = len(dataloader.dataset)
    model.eval()
    value_loss, policy_loss, correct_winner, correct_move = 0, 0, 0, 0
    with torch.no_grad():
        for occupancies, children_visits, values in dataloader:
            pred_values, pred_children_visits = model(occupancies)         
            value_loss += nn.CrossEntropyLoss()(pred_values, values).item()
            policy_loss += nn.CrossEntropyLoss()(pred_children_visits, children_visits).item()
            correct_winner += (pred_values.argmax(1) == values.argmax(1)).type(torch.float).sum().item()
            correct_move += (pred_children_visits.argmax(1) == children_visits.argmax(1)).type(torch.float).sum().item()

    value_loss /= size
    policy_loss /= size
    correct_winner /= size
    correct_move /= size
    value_losses.append(value_loss)
    policy_losses.append(policy_loss)
    winner_accuracies.append(correct_winner * 100)
    move_accuracies.append(correct_move * 100)
    print(f"Test Error: \n Winner accuracy: {(100*correct_winner):>0.1f}%, \n Move accuracy: {(100*correct_move):>0.1f}%, \n Avg value loss: {value_loss:>8f}, \n Avg policy loss: {policy_loss:>8f} \n")

def run(games_dir, output_dir):
    train_dataset, test_dataset = _load_datasets(games_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    model = NeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    value_losses = []
    policy_losses = []
    winner_accuracies = []
    move_accuracies = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train(train_dataloader, optimizer, model)
        _test(test_dataloader, model, value_losses, policy_losses, winner_accuracies, move_accuracies)
    
    # Plotting the value loss, policy loss, winner accuracy, and move accuracy over epochs side by side
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    axs[0, 0].plot(range(1, epochs + 1), value_losses, label='Value Loss', color='blue')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Value Loss over Time')
    axs[0, 0].legend()
    
    axs[0, 1].plot(range(1, epochs + 1), policy_losses, label='Policy Loss', color='orange')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Policy Loss over Time')
    axs[0, 1].legend()
    
    axs[1, 0].plot(range(1, epochs + 1), winner_accuracies, label='Winner Accuracy', color='green')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy (%)')
    axs[1, 0].set_title('Winner Accuracy over Time')
    axs[1, 0].legend()
    
    axs[1, 1].plot(range(1, epochs + 1), move_accuracies, label='Move Accuracy', color='red')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy (%)')
    axs[1, 1].set_title('Move Accuracy over Time')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Done!")