import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np 
import time
import json
import sys
from multiprocessing import freeze_support

def invert_word(word):
    inv = ""
    for i in word[::-1]:
        if i == "A":
            inv += "a"
        if i == "B":
            inv += "b"
        if i == "a":
            inv += "A"
        if i == "b":
            inv += "B"
    return inv
if __name__ == '__main__':
    freeze_support()
    symbol_to_matrix = {
        "A": lmp.A.convert_to_modulo(2),
        "B": lmp.B.convert_to_modulo(2),
        "a": lmp.a.convert_to_modulo(2),
        "b": lmp.b.convert_to_modulo(2)
    }
    w1 = "aBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaBABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBAB"
    w2 = "ABABAABAABABAABABAABABAABAABABAABABAABAABABAABABAABABAABAABABAABABAABABAABAABABAABABAABABAABAABABAABABAABAABABAABABAABABAABAABABABaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaB"
    A = bf.word_to_matrix(symbol_to_matrix,w1)
    B = bf.word_to_matrix(symbol_to_matrix,w2)
    A1 = bf.word_to_matrix(symbol_to_matrix,invert_word(w1))
    B1 = bf.word_to_matrix(symbol_to_matrix,invert_word(w2))
    new_matrices = {
        "A": A,
        "B": B,
        "a": A1,
        "b": B1
    }
    n = 9
    init = bf.calculate_products(new_matrices,n-1)
    init = {key: value for key, value in init.items() if len(key) == n}
    for i in range(10000):
        init = bf.tiered_sampling(init,[1,1],0,20000,20000)
        print(len(init))
        init = bf.extend_in_all_ways_p(new_matrices,init,1)
        print(len(next(iter(init))),bf.min_degree_in_array(init))

