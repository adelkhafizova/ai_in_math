import sys
sys.path.append('../scripts')
import burau_enchanced as lmp
import free_scripts as bf
from itertools import product
import numpy as np
import time
import json
import multiprocessing as mp
from multiprocessing import freeze_support
from tqdm import tqdm  # Optional: for progress tracking

def is_good_word(mtx, word):
    """Check if a word meets the criteria of being 'good'"""
    t = bf.largest_power_range(bf.word_to_matrix(mtx, word))
    return t < 5

def generate_valid_prefixes(prefix_length):
    """Generate all valid prefixes of given length"""
    prefixes = []
    
    def build(prefix="", prev="", remaining=prefix_length):
        if remaining == 0:
            prefixes.append(prefix)
            return
        for g in ['A', 'a', 'B', 'b']:
            if prefix == "" or not ((prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]):
                build(prefix + g, g, remaining - 1)
    
    build('', '', prefix_length)
    return prefixes

def process_chunk(args):
    """Process a chunk of words and return the good ones"""
    mtx, prefix, remaining_length = args
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
    
    # Starting from the given prefix
    last_char = prefix[-1] if prefix else ""
    build(prefix, last_char, remaining_length)
    
    return good_words

def generate_good_words_parallel(mtx, length, num_cores=None):
    """Generate good words of the specified length using multiple cores"""
    if num_cores is None:
        num_cores = mp.cpu_count()
    
    print(f"Using {num_cores} CPU cores")
    
    # Calculate optimal prefix length to get more tasks than cores
    # For better CPU utilization, we want significantly more tasks than cores
    total_tasks_target = num_cores * 8  # Aim for 8x more tasks than cores
    
    # Determine prefix length that gives us enough tasks
    prefix_length = 1
    while 4 * (3 ** (prefix_length - 1)) < total_tasks_target and prefix_length < length:
        prefix_length += 1
    
    print(f"Using prefix length of {prefix_length} for task distribution")
    
    # Generate all valid prefixes of the calculated length
    prefixes = generate_valid_prefixes(prefix_length)
    print(f"Created {len(prefixes)} tasks from prefixes")
    
    # Prepare arguments for each worker
    tasks = [(mtx, prefix, length - len(prefix)) for prefix in prefixes]
    
    # Use a pool of workers with a chunksize appropriate for the number of tasks
    good_words = []
    with mp.Pool(processes=num_cores) as pool:
        # Dynamic chunking to better balance the load
        chunksize = max(1, len(tasks) // (num_cores * 4))
        
        # Optional: use tqdm for progress monitoring
        try:
            from tqdm import tqdm
            results = list(tqdm(pool.imap(process_chunk, tasks, chunksize=chunksize), 
                               total=len(tasks), 
                               desc="Processing word prefixes"))
        except ImportError:
            results = pool.map(process_chunk, tasks, chunksize=chunksize)
        
        for result in results:
            good_words.extend(result)
    
    return good_words

if __name__ == '__main__':
    freeze_support()
    mod = 2
    n = 15  # Word length
    
    # Use all available CPU cores
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
