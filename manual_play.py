def select_move(state, player):
    piece_squares = int(input("How many squares is your piece: "))

    valid_piece_indexes = set()
    for move in piece_moves:
        if state.is_valid_move(player, move) and len(ALL_PIECES[move.piece_index]) != piece_squares:
            valid_piece_indexes.add(move.piece_index)
    
    for piece_index in sorted(valid_piece_indexes):
        print(f"Piece {piece_index}")
        move = piece_moves[piece_index][0]
        new_state = State()
        new_state.play_move(player, move)
        new_state.pretty_print_board(5)
    
    piece_index = int(input("Which piece to play: "))
    valid_moves = [move for move in piece_moves[piece_index] if state.is_valid_move(player, move)]
    if len(valid_moves) == 0:
        return False

    for move_index in range(len(valid_moves)):
        move = valid_moves[move_index]
        new_state = state.copy()
        new_state.play_move(player, move)
        print(f"Move {move_index}")
        new_state.pretty_print_board()

    move_index = int(input("Which move: "))
    return valid_moves[move_index]