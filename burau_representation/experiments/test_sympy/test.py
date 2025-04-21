import sys
sys.path.append('.../scripts')
import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np
import time
import json
import multiprocessing as mp
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    matrices = {
        "A": lmp.A,
        "B": lmp.B,
        "a": lmp.a,
        "b": lmp.b
    }
