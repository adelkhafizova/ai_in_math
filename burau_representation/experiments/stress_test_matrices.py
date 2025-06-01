import sys
sys.path.append('../scripts')
import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np 
import time
import json
import sys
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    c = bf.generate_reduced_words(5,generators = ["a","b"])
    print(c)
    mod = 2
    n = 9
    symbol_to_matrix = {
        "A": lmp.A.convert_to_modulo(mod),
        "B": lmp.B.convert_to_modulo(mod),
        "a": lmp.a.convert_to_modulo(mod),
        "b": lmp.b.convert_to_modulo(mod)
    }
    a = {"A":None,"B":None,"a":None,"b":None}
    print(bf.tiered_word_sampling_p(symbol_to_matrix,a,[1,1,0.5,0.2],0,30000,30000))

    for i in range(10000):
        c = bf.tiered_word_sampling_p(symbol_to_matrix,c,[1,1,1,0.2],0,1000,5000)
        print(len(c))
        c = bf.extend_word_in_all_ways_p(c,1)

