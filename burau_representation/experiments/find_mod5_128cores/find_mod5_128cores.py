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
    mod = 5
    n = 13
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


    
    for i in range(20000):
        tier_percentages = [1,1,1]
        remaining_percentage = 0
        min_num = 1000000
        max_num = 2000000
        c = bf.tiered_sampling(c,tier_percentages,remaining_percentage,min_num,max_num)
        if(n%10 == 0):
            #plotting
            degrees =bf.get_invariant_picture(c,bf.largest_power_range)
            keys = list(degrees.keys())
            values = list(degrees.values())
            plt.clf()
            plt.bar(keys, values)
            for x, y in zip(keys, values):
                plt.text(x, y + 0.2, str(y), ha='center', va='bottom')
            plt.xlabel('Degree')
            plt.ylabel('Number of words')
            plt.title(f"Histogram for length n = {len(next(iter(c)))} with parameters {tier_percentages,min_num,max_num}")
            plt.savefig(f"plotsmod{mod}/histogram_{len(next(iter(c)))}.png")
            #plotting
            n += 1

        c = bf.extend_in_all_ways_p(matrices_mod,c,1)
        print(len(next(iter(c))),bf.min_invariant_in_array(c,bf.largest_power_range))