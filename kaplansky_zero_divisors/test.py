import networkx as nx
import numpy as np
from taiko import *


def main():
    taiko_example_1 = [(4, 8, 5, 5), (6, 4, 7, 3), (1, 1, 2, 9), (6, 5, 7, 4),
                       (1, 2, 2, 1), (5, 6, 7, 5), (1, 3, 3, 1), (4, 9, 5, 8),
                       (6, 7, 7, 6), (1, 4, 2, 3), (3, 2, 4, 1), (6, 8, 7, 7),
                       (1, 5, 2, 4), (4, 2, 7, 8), (1, 6, 3, 4), (4, 3, 5, 9),
                       (1, 7, 2, 6), (3, 5, 4, 4), (5, 1, 6, 9), (1, 8, 2, 7),
                       (4, 5, 5, 2), (6, 1, 7, 9), (2, 2, 3, 9), (5, 7, 6, 6),
                       (2, 5, 3, 3), (2, 8, 3, 6), (6, 2, 7, 1), (3, 7, 4, 6),
                       (5, 3, 7, 2), (3, 8, 4, 7), (5, 4, 6, 3)]
    taiko_example_2 = [(1, 1, 2, 2), (1, 2, 2, 3), (2, 1, 3, 3), (4, 1, 1, 3),
                       (3, 1, 5, 4), (3, 2, 4, 4), (1, 4, 6, 5), (2, 4, 7, 3),
                       (3, 4, 6, 3), (3, 5, 4, 2), (4, 3, 7, 5), (5, 1, 1, 5),
                       (5, 2, 7, 4), (5, 3, 8, 5), (8, 1, 2, 5), (8, 4, 4, 5),
                       (5, 5, 7, 2), (6, 1, 8, 2), (6, 2, 8, 3), (7, 1, 6, 4)]

    taiko = Taiko(taiko_example_2)


if __name__ == '__main__':
    main()
