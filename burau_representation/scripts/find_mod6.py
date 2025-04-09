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
    symbol_to_matrix = {
        "A": lmp.A.convert_to_modulo(2),
        "B": lmp.B.convert_to_modulo(2),
        "a": lmp.a.convert_to_modulo(2),
        "b": lmp.b.convert_to_modulo(2)
    }

    print(bf.word_to_matrix(symbol_to_matrix,"aBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaBABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBAB"))
    print(bf.word_to_matrix(symbol_to_matrix,"ABABAABAABABAABABAABABAABAABABAABABAABAABABAABABAABABAABAABABAABABAABABAABAABABAABABAABABAABAABABAABABAABAABABAABABAABABAABAABABABaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBBaBaBBaBBaBaBBaBaBBaBBaB"))



