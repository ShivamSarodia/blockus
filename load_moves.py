import pickle
from moves import Move

def load_moves():
    with open("moves.pickle", "rb") as f:
        return pickle.load(f)