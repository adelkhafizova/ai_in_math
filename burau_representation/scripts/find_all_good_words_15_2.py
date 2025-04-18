import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np 
import time
import json
import sys
from multiprocessing import freeze_support

def generate_good_words(mtx,length):
    good = []
    def build(word, prev, remaining):
        if remaining == 0:
            t = bf.largest_power_range(bf.word_to_matrix(mtx,word))
            if t<5:
                good.append(word)
            return
        for g in ['A', 'a', 'B', 'b']:
            if (prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]:
                continue
            build(word + g, g, remaining - 1)

    build('', '', length)
    return good

if __name__ == '__main__':
    freeze_support()
    c = bf.generate_reduced_words(5,generators = ["a","b"])
    print(c)
    mod = 2
    n = 15
    symbol_to_matrix = {
        "A": lmp.A.convert_to_modulo(mod),
        "B": lmp.B.convert_to_modulo(mod),
        "a": lmp.a.convert_to_modulo(mod),
        "b": lmp.b.convert_to_modulo(mod)
    }
    good = generate_good_words(symbol_to_matrix,n)
    print(len(good))
