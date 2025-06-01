import burau_enchanced as lmp
import ai_in_math.burau_representation.scripts.old.free_scripts as bf
from itertools import product
import numpy as np 
import json
import sys
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    mod = 5
    n = 10
    symbol_to_matrix = {
        "A": lmp.A,
        "B": lmp.B,
        "a": lmp.a,
        "b": lmp.b
    }
    a = bf.calculate_products(symbol_to_matrix,n-1)
    print(len(a))
    modu = {key: value.convert_to_modulo(mod) for key, value in a.items()}
    matrices_mod = {
        "A": lmp.A.convert_to_modulo(mod),
        "B": lmp.B.convert_to_modulo(mod),
        "a": lmp.a.convert_to_modulo(mod),
        "b": lmp.b.convert_to_modulo(mod)
    }
    b = {key: value for key, value in modu.items() if len(key) == n}  
    c = b
    a.clear()
    for i in range(10000):
        c = bf.tiered_sampling(c,[1,1,0.5,0.2],0,50000)
        print(len(c))
        c = bf.extend_in_all_ways_p(matrices_mod,c,1)
        print(len(next(iter(c))),bf.min_degree_in_array(c))
