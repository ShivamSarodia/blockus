BOARD_SIZE = 20

class Move:
    def __init__(self, piece_index, new_occupied, display_coordinates):
        self.piece_index = piece_index
        self.new_occupied = new_occupied
        self.new_corners = None
        self.new_adjacents = None
        self.display_coordinates = display_coordinates

    def populate_new_corners(self):
        self.new_corners = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for x_to_check in [x + 1, x - 1]:
                    for y_to_check in [y + 1, y - 1]:
                        if x_to_check < 0 or x_to_check >= BOARD_SIZE or y_to_check < 0 or y_to_check >= BOARD_SIZE:
                            continue
                        if self.new_occupied & (1 << (x_to_check + y_to_check * BOARD_SIZE)):
                            self.new_corners |= 1 << (x + y * BOARD_SIZE)

    def populate_new_adjacents(self):
        self.new_adjacents = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for x_to_check, y_to_check in [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]:
                    if x_to_check < 0 or x_to_check >= BOARD_SIZE or y_to_check < 0 or y_to_check >= BOARD_SIZE:
                        continue
                    if self.new_occupied & (1 << (x_to_check + y_to_check * BOARD_SIZE)):
                        self.new_adjacents |= 1 << (x + y * BOARD_SIZE)        

    def __hash__(self) -> int:
        return hash(self.new_occupied)

    def __eq__(self, o: object) -> bool:
        return self.new_occupied == o.new_occupied