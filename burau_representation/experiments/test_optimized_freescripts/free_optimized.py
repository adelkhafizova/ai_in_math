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

def process_batch(args):
    """Process a batch of entries in one process with improved memory efficiency"""
    dict_ref, batch_entries, t = args
    
    if t == 0:
        return batch_entries
    
    symbols = ["A", "B", "a", "b"]
    results = {}
    
    for key, value in batch_entries.items():
        # Get last character for swapcase check
        last_char = key[-1] if key else None
        
        # Build set of allowed symbols more efficiently
        if last_char:
            last_swapped = last_char.swapcase()
            allowed_symbols = [s for s in symbols if s != last_swapped]
        else:
            allowed_symbols = symbols
            
        # Create new entries
        for i in allowed_symbols:
            if i in dict_ref:  # Ensure symbol exists in dict_ref
                results[key+i] = value * dict_ref[i]
    
    # Free memory
    batch_entries = None
    gc.collect()
    
    return results

def chunked_dict_items(data_dict, chunk_size):
    """Yield chunks of dictionary items without creating full list"""
    items_iter = iter(data_dict.items())
    while True:
        # Get next chunk
        chunk = dict(islice(items_iter, chunk_size))
        if not chunk:
            break
        yield chunk

def extend_in_all_ways_p(dict_ref, entries, t, max_memory_ratio=0.75):
    """
    Extend entries by multiplying with dictionary values, dividing work into
    equal parts processed in parallel with memory optimization.
    
    Args:
        dict_ref: Reference dictionary mapping symbols to values
        entries: Dictionary of entries to extend
        t: Number of iterations to perform
        max_memory_ratio: Maximum ratio of entries to process at once (0-1)
    """
    if t == 0 or not entries:
        return entries
    
    # Get available CPU count but limit based on problem size and memory concerns
    available_cpus = multiprocessing.cpu_count()
    entries_size = len(entries)
    
    # For small datasets, process sequentially to avoid overhead
    if available_cpus <= 1 or entries_size < 1000:
        return extend_in_all_ways_sequential(dict_ref, entries, t)
    
    # Determine optimal number of processes based on data size
    optimal_processes = min(
        available_cpus,
        max(1, int(entries_size / 1000))  # At least 1000 items per process
    )
    
    # Calculate batch size with memory concerns in mind
    batch_size = max(1, int(entries_size * max_memory_ratio / optimal_processes))
    
    try:
        results = {}
        with multiprocessing.Pool(processes=optimal_processes) as pool:
            # Process batches one at a time to control memory usage
            for batch in chunked_dict_items(entries, batch_size):
                # Submit batch for processing
                batch_results = pool.apply(process_batch, [(dict_ref, batch, 1)])
                
                # Collect results
                if batch_results:
                    results.update(batch_results)
                
                # Clear batch to free memory
                batch = None
                gc.collect()
        
        # Free original entries
        entries = None
        gc.collect()
        
        # Continue extending with the combined results
        if t > 1:
            return extend_in_all_ways_p(dict_ref, results, t-1, max_memory_ratio)
        else:
            return results
        
    except Exception as e:
        print(f"Parallel processing failed: {e}. Falling back to sequential.")
        # Fall back to sequential processing
        return extend_in_all_ways_sequential(dict_ref, entries, t)

def extend_in_all_ways_sequential(dict_ref, entries, t):
    """Sequential version of the extension function for fallback"""
    if t == 0 or not entries:
        return entries
        
    symbols = ["A", "B", "a", "b"]
    results = {}
    
    for key, value in entries.items():
        # Get last character for swapcase check
        last_char = key[-1] if key else None
        
        # Build list of allowed symbols more efficiently
        if last_char:
            last_swapped = last_char.swapcase()
            allowed_symbols = [s for s in symbols if s != last_swapped]
        else:
            allowed_symbols = symbols
            
        # Create new entries
        for i in allowed_symbols:
            if i in dict_ref:  # Ensure symbol exists in dict_ref
                results[key+i] = value * dict_ref[i]
    
    # Free memory
    entries = None
    gc.collect()
    
    # Continue with next iteration
    if t > 1:
        return extend_in_all_ways_sequential(dict_ref, results, t-1)
    else:
        return results


def tiered_sampling(results, tier_percentages=[], remaining_percentage=0, min_num=0, max_num=1000000000, invariant=None):
    """
    Sample from different tiers of results based on specified percentages.
    
    Args:
        results: Dictionary with keys and their associated values to evaluate
        tier_percentages: List of percentages for each tier, starting from best tier
        remaining_percentage: Percentage to sample from items not in any tier
        min_num: Minimum number of results to return
        max_num: Maximum number of results to return
        invariant: Function to calculate invariants for sorting (default: None)
    
    Returns:
        Dictionary containing the sampled items from each tier
    """
    
    if not results or max_num <= 0:
        return {}
    
    if not tier_percentages:
        # If no tiers specified, just return random sample up to max_num
        return dict(random.sample(list(results.items()), min(len(results), max_num)))
    
    # Calculate invariants without storing all of them at once
    invariant_to_keys = defaultdict(list)
    for key, value in results.items():
        if invariant:
            inv_value = invariant(value)
            invariant_to_keys[inv_value].append(key)
    
    # Get sorted invariant values (we only need the unique values, not all entries)
    sorted_invariants = sorted(invariant_to_keys.keys())
 
    final = {}
    remaining_slots = max_num
    
    # Process each tier
    for i, percentage in enumerate(tier_percentages):
        if percentage <= 0 or remaining_slots <= 0:
            continue
            
        # Get keys for the current tier (if available)
        if i < len(sorted_invariants):
            current_inv = sorted_invariants[i]
            tier_keys = invariant_to_keys[current_inv]
            
            # Calculate how many items to sample
            sample_size = min(int(percentage * len(tier_keys)), remaining_slots)
            
            if sample_size > 0:
                # Random sampling without creating a full list of items
                if sample_size == len(tier_keys):
                    # Take all items
                    sampled_keys = tier_keys
                else:
                    # Random sample
                    sampled_keys = random.sample(tier_keys, sample_size)
                
                # Add sampled items to final result
                for key in sampled_keys:
                    final[key] = results[key]
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
    
    # Fill to minimum if needed
    if len(final) < min_num and min_num <= len(results):
        # Get keys that weren't sampled yet
        remaining_keys = [k for k in results if k not in final]
        
        # Calculate how many more items we need
        additional_needed = min_num - len(final)
        
        # Sample additional items to meet minimum
        if additional_needed > 0:
            additional_samples = random.sample(remaining_keys, min(additional_needed, len(remaining_keys)))
            for key in additional_samples:
                final[key] = results[key]
    
    return final