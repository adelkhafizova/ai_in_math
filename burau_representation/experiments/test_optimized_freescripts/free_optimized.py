import sys
sys.path.append('../../scripts')
import burau_enchanced as lmp
from itertools import product
from collections import defaultdict
import numpy as np 
import math
import json
from tqdm import tqdm
import random
import multiprocessing
from itertools import islice
import gc

matrices_mod = {
        "A": lmp.A.convert_to_modulo(2),
        "B": lmp.B.convert_to_modulo(2),
        "a": lmp.a.convert_to_modulo(2),
        "b": lmp.b.convert_to_modulo(2)
    }
symbols = ["A", "B", "a", "b"]
def extend_word(pair_word_matrix):
    results = dict()
    for i in symbols:
        if i != pair_word_matrix[0][-1].swapcase():  # Only exclude the one that matches
            results[pair_word_matrix[0] + i] = pair_word_matrix[1] * matrices_mod[i]
    return results

def extend_in_all_ways_p(dict_ref, entries, t):
    """
    Extend entries by multiplying with dictionary values, using a non-recursive approach
    with a single multiprocessing pool.
    """
    current_entries = entries
    
    # Create a single pool for all iterations
    with multiprocessing.Pool() as pool:
        for _ in range(t):
            if not current_entries:
                break
                
            list_of_entries = []
            while current_entries:
                k, v = current_entries.popitem()
                list_of_entries.append([k, v])
            
            # Process all entries in parallel
            batch_results = pool.map(extend_word, list_of_entries)
            
            # Combine results
            current_entries = {}
            for triple in batch_results:
                current_entries.update(triple)
            
            # Optional: Force garbage collection to free memory
            gc.collect()
    
    return current_entries
        

