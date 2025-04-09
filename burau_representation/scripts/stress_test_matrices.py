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
    mod = 5
    n = 11
    symbol_to_matrix = {
        "A": lmp.A,
        "B": lmp.B,
        "a": lmp.a,
        "b": lmp.b
    }
    a = bf.extend_in_all_ways_p(symbol_to_matrix,symbol_to_matrix,9)
    print(len(a))
