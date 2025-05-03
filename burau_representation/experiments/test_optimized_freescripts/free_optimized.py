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
    Extend entries by multiplying with dictionary values, dividing work into
    equal parts processed in parallel.
    """
    
    if t == 0:
        return entries
        
    
    
    list_of_entries = []

    while entries:
        k, v = entries.popitem()
        list_of_entries.append([k, v])

    p = multiprocessing.Pool()
    batch_results = p.map(extend_word, list_of_entries)
    p.close()
    p.join()
    combined_results = {}
    while batch_results:
        triple = batch_results.pop()
        combined_results.update(triple)
    # Continue extending with the combined results
    return extend_in_all_ways_p(dict_ref, combined_results, t-1)
        

