import pickle
from moves import Move

BOARD_SIZE = 20
ALL_PIECES = [
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

def generate_moves(): 
    piece_moves = []

    # For each piece, generate all possible moves of that piece (assuming player 4 is making the move, so we can && it easily).
    # Map from piece number -> list of moves
    for piece_index in range(len(ALL_PIECES)):
        moves_for_piece = set()

        for rotation in range(4):
            for flip in range(2):
                piece = ALL_PIECES[piece_index][:]

                # Apply the rotation to the piece
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

                # Create the occupation string.
                for x in range(BOARD_SIZE - max_x):
                    for y in range(BOARD_SIZE - max_y):
                        occupation_bitstring = 0
                        display_coordinates = []
                        for piece_x, piece_y in piece:
                            display_coordinates.append((x + piece_x, y + piece_y))
                            occupation_bitstring |= 1 << ((x + piece_x + (y + piece_y) * BOARD_SIZE))
                
                        moves_for_piece.add(Move(piece_index, occupation_bitstring, display_coordinates))
                
        for move in moves_for_piece:
            move.populate_new_corners()
            move.populate_new_adjacents()

        piece_moves.append(moves_for_piece)


    for piece_index in range(len(piece_moves)):
        piece_moves[piece_index] = sorted(piece_moves[piece_index], key=lambda x: x.new_occupied)

    return piece_moves

if __name__ == "__main__":
    piece_moves = generate_moves()
    with open("moves.pickle", "wb") as f:
        pickle.dump(piece_moves, f)
