import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', type=int, required=True)
parser.add_argument('--precompute_directory', type=str, required=True)
args = parser.parse_args()

BOARD_SIZE = args.board_size
PRECOMPUTE_DIRECTORY = args.precompute_directory
PIECES = [
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
    [(1, 1), (0, 1), (1, 0), (2, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (1, 0), (2, 0), (0, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    [(0, 0), (1, 0), (2, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0)],
]

output = {}

print("Populating values...")

output["new_occupieds"] = []
output["piece_indices"] = []
output["new_corners"] = []
output["new_adjacents"] = []

for piece_index in tqdm(range(len(PIECES))):
    occupieds_for_piece = []

    for rotation in range(4):
        for flip in range(2):
            piece = PIECES[piece_index][:]

            for _ in range(rotation + 1):
                piece = [(y, -x) for x, y in piece]

            # Apply the flip to the piece
            if flip:
                piece = [(x, -y) for x, y in piece]

            # Find the bottom left of the piece now.
            min_x = min(x for x, y in piece)
            min_y = min(y for x, y in piece)

            # Shift the piece so that the bottom left corner is at (0, 0)
            piece = [(x - min_x, y - min_y) for x, y in piece]

            # Find the maximum x and y values of the piece
            max_x = max(x for x, y in piece)
            max_y = max(y for x, y in piece)

            for x in range(BOARD_SIZE - max_x):
                for y in range(BOARD_SIZE - max_y):
                    occupieds_for_piece.append(np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool))
                    for piece_x, piece_y in piece:
                        occupieds_for_piece[-1][x + piece_x][y + piece_y] = True

    occupieds_for_piece = np.unique(np.array(occupieds_for_piece, dtype=bool), axis=0)
    output["new_occupieds"].append(occupieds_for_piece)
    output["piece_indices"] += [piece_index] * occupieds_for_piece.shape[0]

    for i, new_occupieds in enumerate(occupieds_for_piece):
        new_corners = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        new_adjacents = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=bool)

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                # Populate new corners
                for x_to_check in [x + 1, x - 1]:
                    for y_to_check in [y + 1, y - 1]:
                        if x_to_check < 0 or x_to_check >= BOARD_SIZE or y_to_check < 0 or y_to_check >= BOARD_SIZE:
                            continue
                        if new_occupieds[x_to_check, y_to_check]:
                            new_corners[0][x, y] = True

                # Populate new adjacents
                for x_to_check, y_to_check in [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]:
                    if x_to_check < 0 or x_to_check >= BOARD_SIZE or y_to_check < 0 or y_to_check >= BOARD_SIZE:
                        continue
                    if new_occupieds[x_to_check, y_to_check]:
                        new_adjacents[0][x, y] = True

        output["new_corners"].append(new_corners)
        output["new_adjacents"].append(new_adjacents)

output["new_occupieds"] = np.concatenate(output["new_occupieds"])
NUM_MOVES = output["new_occupieds"].shape[0]

output["scores"] = np.sum(output["new_occupieds"], axis=(1, 2), dtype=int)
assert output["scores"].shape == (NUM_MOVES,)
output["piece_indices"] = np.array(output["piece_indices"], dtype=int)
output["new_corners"] = np.concatenate(output["new_corners"])
output["new_adjacents"] = np.concatenate(output["new_adjacents"])

print("Done populating moves dictionary.") 

print(f"Generated {NUM_MOVES} moves.")

print("Computing moves using same piece...")
moves_using_same_piece = output["piece_indices"][:, np.newaxis] == output["piece_indices"]
assert moves_using_same_piece.shape == (NUM_MOVES, NUM_MOVES)

def batch_and(a1, a2):
    batch_size = 2000
    result = np.zeros((NUM_MOVES, NUM_MOVES), dtype=bool)
    for i in tqdm(range(0, NUM_MOVES, batch_size), leave=False):
        end_i = min(i + batch_size, NUM_MOVES)
        new_a1_batch = a1[i:end_i]  # Shape: (batch_size_i, 20, 20)
        new_a1_batch_expanded = new_a1_batch[:, np.newaxis, :, :]  # Shape: (batch_size_i, 1, 20, 20)

        for j in tqdm(range(0, NUM_MOVES, batch_size), leave=False):
            end_j = min(j + batch_size, NUM_MOVES)
            new_a2_batch = a2[j:end_j]  # Shape: (batch_size_j, 20, 20)

            # Compute the boolean operation in the current batch
            temp_result = (new_a1_batch_expanded & new_a2_batch).any(axis=(2, 3))
            result[i:end_i, j:end_j] = temp_result

    return result

print("Computing moves occupying new adjacents...")
moves_occupying_new_adjacents = batch_and(output["new_adjacents"], output["new_occupieds"])
assert moves_occupying_new_adjacents.shape == (NUM_MOVES, NUM_MOVES)

print("Computing moves occupying new occupieds...")
moves_occupying_new_occupieds = batch_and(output["new_occupieds"], output["new_occupieds"])
assert moves_occupying_new_adjacents.shape == (NUM_MOVES, NUM_MOVES)

print("Computing moves occupying new corners...")
moves_occupying_new_corners = batch_and(output["new_corners"], output["new_occupieds"])
assert moves_occupying_new_corners.shape == (NUM_MOVES, NUM_MOVES)

print("Computing output values...")

output["moves_ruled_out_for_player"] = moves_using_same_piece | moves_occupying_new_adjacents
del moves_using_same_piece
del moves_occupying_new_adjacents

output["moves_ruled_out_for_all"] = moves_occupying_new_occupieds
output["moves_enabled_for_player"] = moves_occupying_new_corners

# Save each of the arrays to disk.
print("Saving outputs to disk...")
os.makedirs(PRECOMPUTE_DIRECTORY, exist_ok=True)
for key, value in output.items():
    np.save(f"{PRECOMPUTE_DIRECTORY}/{key}.npy", value)

print(f"Done.")