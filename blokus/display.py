import numpy as np
from matplotlib import pyplot as plt

from moves import BOARD_SIZE

class Display:
    def __init__(self, occupancies, overlay_dots=None):
        self.occupancies = occupancies
        self.overlay_dots = overlay_dots

    def show(self):
        grid = np.zeros((BOARD_SIZE, BOARD_SIZE, 3))
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.occupancies[0, x, y]:
                    color = [1, 0, 0]
                elif self.occupancies[1, x, y]:
                    color = [0, 1, 0]
                elif self.occupancies[2, x, y]:
                    color = [0.25, 0.25, 1]
                elif self.occupancies[3, x, y]:
                    color = [0, 1, 1]
                else:
                    color = [1, 1, 1]
                grid[y, x] = color

        # Plot the grid
        plt.imshow(grid, interpolation='nearest')
        plt.axis('on')  # Show the axes
        plt.grid(color='black', linestyle='-', linewidth=2)  # Add gridlines

        if self.overlay_dots is not None:
            x_coords, y_coords = [], []
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    if self.overlay_dots[x, y]:
                        x_coords.append(x)
                        y_coords.append(y)

            plt.scatter(x_coords, y_coords, color='black', s=20)  # Draw black dots

        # Adjust the gridlines to match the cells
        plt.xticks(np.arange(-0.5, BOARD_SIZE, 1), [])
        plt.yticks(np.arange(-0.5, BOARD_SIZE, 1), [])
        plt.gca().set_xticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.show()