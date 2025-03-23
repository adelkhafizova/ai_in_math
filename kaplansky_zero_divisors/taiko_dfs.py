from collections import deque
from itertools import permutations
import networkx as nx
from taiko import *

max_M = 4
max_N = 4
do_BFS = False
max_depth = 3

def main():
    stack = deque()
    stack.append(tuple())
    explored = set()
    leaves = []
    max_num_two_cells = 0
    best_ratio = 0

    while stack:
        current = stack.pop()
        if current not in explored:
            explored.add(current)
            current_two_cell_list = list(current)
            current_taiko = Taiko(current_two_cell_list)
            can_taiko_be_extended = False
            i1, j1 = min(current_taiko.available_edges)
            for i2, j2 in tuple(current_taiko.available_edges):
                if current_taiko.can_add_two_cell(i1, j1, i2, j2, max_M, max_N):
                    if current_taiko.add_two_cell(i1, j1, i2, j2):
                        if current_taiko.no_fold() and (current_taiko.is_girth_p_q(6, 3)):
                            can_taiko_be_extended = True
                            neighbor = current + ((i1, j1, i2, j2),)
                            stack.append(neighbor)
                        current_taiko.pop_two_cell()

                elif current_taiko.can_add_two_cell(i2, j2, i1, j1, max_M, max_N):
                    if current_taiko.add_two_cell(i2, j2, i1, j1):
                        if current_taiko.no_fold() and (current_taiko.is_girth_p_q(6, 3)):
                            can_taiko_be_extended = True
                            neighbor = current + ((i2, j2, i1, j1),)
                            stack.append(neighbor)
                        current_taiko.pop_two_cell()
            if not can_taiko_be_extended:
                # leaves.append(current)
                num_two_cells = len(current)
                if num_two_cells > max_num_two_cells:
                    max_num_two_cells = num_two_cells

                if 2 * num_two_cells / (current_taiko.M * current_taiko.N) > best_ratio:
                    best_ratio = 2 * num_two_cells / (current_taiko.M * current_taiko.N)
    print(max_num_two_cells)
    print(best_ratio)


if __name__ == '__main__':
    main()
