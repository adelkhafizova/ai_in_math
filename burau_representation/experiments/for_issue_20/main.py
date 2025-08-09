import sys
sys.path.append('../../scripts')
import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np 
import matplotlib.pyplot as plt
import json
import sys
from multiprocessing import freeze_support


 

if __name__ == '__main__':
    freeze_support()
    def get_beans(invariant_dictionary):
        beans = [0]*len(invariant_dictionary)
        for key,value in invariant_dictionary.items():
            beans[key] = key
        beans = [x for x in beans if invariant_dictionary[key] != 0]
        return beans
   
    mod = 2
    n = 5
    symbol_to_matrix = {
        "A": lmp.A,
        "B": lmp.B,
        "a": lmp.a,
        "b": lmp.b
    }
    a = bf.calculate_products(symbol_to_matrix,n-1)
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

    beans_for_iter = []
    good_words = []
    for i in range(15):
        tier_percentages = [1,1,1]
        remaining_percentage = 0
        min_num = 0
        max_num = 5000
        c = bf.tiered_sampling(c,tier_percentages,remaining_percentage,min_num,max_num)
        beans_for_iter.append(get_beans(bf.get_invariant_picture(c,bf.largest_power_range)))
        c = bf.extend_in_all_ways_p(matrices_mod,c,1)

        print(len(next(iter(c))),bf.min_invariant_in_array(c,bf.largest_power_range))
        good_words.append()
    print(beans_for_iter)