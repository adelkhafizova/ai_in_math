import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np
import time
import json
import sys
import multiprocessing as mp
from multiprocessing import freeze_support

def is_good_word(mtx, word):
    """Check if a word meets the criteria of being 'good'"""
    t = bf.largest_power_range(bf.word_to_matrix(mtx, word))
    return t < 5

def process_chunk(args):
    """Process a chunk of words and return the good ones"""
    mtx, prefix, length = args
    good_words = []
    
    def build(word, prev, remaining):
        if remaining == 0:
            if is_good_word(mtx, word):
                good_words.append(word)
            return
        for g in ['A', 'a', 'B', 'b']:
            if (prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]:
                continue
            build(word + g, g, remaining - 1)
    
    if prefix:
        # Continue building from the prefix
        build(prefix, prefix[-1], length - len(prefix))
    else:
        # Start from scratch
        build('', '', length)
    
    return good_words

def generate_good_words_parallel(mtx, length, num_cores=None):
    """Generate good words of the specified length using multiple cores"""
    if num_cores is None:
        num_cores = mp.cpu_count()
    
    # For small lengths, just use the sequential version
    if length <= 2:
        return generate_good_words(mtx, length)
    
    # Generate prefixes to distribute work
    prefixes = []
    prefix_length = min(2, length)
    
    def generate_valid_prefixes(prefix="", prev="", remaining=prefix_length):
        if remaining == 0:
            prefixes.append(prefix)
            return
        for g in ['A', 'a', 'B', 'b']:
            if prefix == "" or not ((prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]):
                generate_valid_prefixes(prefix + g, g, remaining - 1)
    
    generate_valid_prefixes()
    
    # Prepare arguments for each worker
    tasks = [(mtx, prefix, length) for prefix in prefixes]
    
    # Start the worker pool and process chunks
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, tasks)
    
    # Combine results from all workers
    good_words = []
    for result in results:
        good_words.extend(result)
    
    return good_words

def generate_good_words(mtx, length):
    """Sequential version of good words generator"""
    good = []
    def build(word, prev, remaining):
        if remaining == 0:
            t = bf.largest_power_range(bf.word_to_matrix(mtx, word))
            if t < 5:
                good.append(word)
            return
        for g in ['A', 'a', 'B', 'b']:
            if (prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]:
                continue
            build(word + g, g, remaining - 1)

    build('', '', length)
    return good

if __name__ == '__main__':
    freeze_support()
    mod = 2
    n = 15
    num_cores = mp.cpu_count()
    
    start_time = time.time()
    
    symbol_to_matrix = {
        "A": lmp.A.convert_to_modulo(mod),
        "B": lmp.B.convert_to_modulo(mod),
        "a": lmp.a.convert_to_modulo(mod),
        "b": lmp.b.convert_to_modulo(mod)
    }
    
    good = generate_good_words_parallel(symbol_to_matrix, n, num_cores)
    
    end_time = time.time()
    print(f"Found {len(good)} good words of length {n}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")