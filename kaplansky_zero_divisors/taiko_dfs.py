from collections import deque
from itertools import permutations
import networkx as nx
from taiko import *


def main():
    stack = deque()
    stack.append(frozenset())
    explored = set()
    leaves = []
    max_M = 4
    max_N = 4

    while stack:
        current = stack.pop()
        if current not in explored:
            explored.add(current)
            current_two_cell_list = list(current)
            current_taiko = Taiko(current_two_cell_list)
            m = current_taiko.M
            n = current_taiko.N
            can_taiko_be_extended = False
            for i1, i2 in permutations(range(1, m + 3), 2):
                for j1, j2 in permutations(range(1, n + 3), 2):
                    if current_taiko.can_add_two_cell(i1, j1, i2, j2, max_M, max_N):
                        current_taiko.add_two_cell(i1, j1, i2, j2)
                        if current_taiko.no_fold() and current_taiko.is_girth_p_q(4, 4):
                            can_taiko_be_extended = True
                            neighbor = current.union(frozenset(((i1, j1, i2, j2),)))
                            stack.append(neighbor)
                        current_taiko.remove_two_cell(i1, j1, i2, j2)
            if not can_taiko_be_extended:
                leaves.append(current)


if __name__ == '__main__':
    main()
