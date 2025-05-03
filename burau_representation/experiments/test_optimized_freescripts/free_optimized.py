import sys
sys.path.append('../../scripts')
import burau_enchanced as lmp
from itertools import product
from collections import defaultdict
import numpy as np 
import multiprocessing
import time
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
    s = time.time()
    # Create a single pool for all iterations
    num_cpus = multiprocessing.cpu_count()
    print(f"Number of cpus:{num_cpus}")
    with multiprocessing.Pool(processes=num_cpus) as pool:
        print(f"Time to create pools:{time.time()-s}")
        for i in range(t):
            print(f"iteration{i}\n")
            if not current_entries:
                break
                
            list_of_entries = []
            s = time.time()
            while current_entries:
                k, v = current_entries.popitem()
                list_of_entries.append([k, v])
            print(f"Time to transform to list:{time.time()-s}")
            
            s = time.time()
            # Process all entries in parallel
            batch_results = pool.map(extend_word, list_of_entries)
            print(f"Time to extend in parralel:{time.time()-s}")

            s = time.time()
            # Combine results
            current_entries = {}
            while batch_results:
               triple = batch_results.pop()
               current_entries.update(triple)
            print(f"Time to combine:{time.time()-s}")
            # Optional: Force garbage collection to free memory
            s = time.time()
            gc.collect()
            print(f"Time to collect garbage:{time.time()-s}")
    
    return current_entries
        

