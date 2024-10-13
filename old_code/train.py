import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', type=int, required=True)
args = parser.parse_args()

BOARD_SIZE = args.board_size

def generate_dataloader(dir, skip_endgame=False):
    # Load boards.npy from gamedata directory.
    print("Loading data...")
    np_boards = np.load(dir + '/boards.npy')
    np_turns = np.load(dir + '/turns.npy')
    np_results = np.load(dir+ '/results.npy')

    if skip_endgame:
        to_include = np.sum(np_boards, axis=(1, 2, 3)) < 35
        np_boards = np_boards[to_include]
        np_turns = np_turns[to_include]
        np_results = np_results[to_include]

    boards = torch.Tensor(np_boards)
    turns = torch.Tensor(np_turns)
    results = torch.Tensor(np_results)

    train_dataset = TensorDataset(boards, turns, results)
    return DataLoader(train_dataset, batch_size=256, shuffle=True)

train_dataloader = generate_dataloader("game_data")
test_dataloader = generate_dataloader("game_data_test")
early_game_test_dataloader = generate_dataloader("game_data_test", skip_endgame=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, boards, turns):
        x = self.flatten(boards)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (boards, turns, results) in enumerate(dataloader):
        # Compute prediction error
        pred = model(boards, turns)
        loss = loss_fn(pred, results)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(boards)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, stop_for_inspect=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for boards, turns, results in dataloader:
            pred = model(boards, turns)
            test_loss += loss_fn(pred, results).item()
            correct += (pred.argmax(1) == results.argmax(1)).type(torch.float).sum().item()
            
            if stop_for_inspect:
                import ipdb; ipdb.set_trace()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    test(early_game_test_dataloader, model, loss_fn)
print("Done!")