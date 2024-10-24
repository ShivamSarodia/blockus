import sys
sys.path = ["/Users/shivamsarodia/Dev/blockus/src"] + sys.path

import numpy as np
from inference.array_queue import ArrayQueue

queue = ArrayQueue(20, [
    np.empty((2, 3, 4), dtype=float),
    np.empty((), dtype=int),
    np.empty((4,), dtype=int),
])

inputs = [
    np.random.rand(2, 3, 4),
    5,
    np.random.randint(0, 10, 4),
]

tensor = queue._to_torch(inputs)
outputs = queue._from_torch(tensor)

print(inputs)
print(outputs)