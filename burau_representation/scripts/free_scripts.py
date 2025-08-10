# import burau_enchanced as lmp
from itertools import product
from collections import defaultdict
import numpy as np 
import math
import json
from tqdm import tqdm
import random
import multiprocessing
from functools import partial
#_p means parralelization
'''symbol_to_matrix = {
    "A": lmp.A,
    "B": lmp.B,
    "a": lmp.a,
    "b": lmp.b
}'''

# Inverse symbol pairs
inverses = {"A": "a", "a": "A", "B": "b", "b": "B"}

def word_to_matrix(dict,word):
    result  = dict[word[0]]
    for i in range(1,len(word)):
        result *= dict[word[i]]
    return result

def largest_power_range_word(dictionary,word):
    """
    Compute the largest positive and negative powers of x for all entries in a LaurentMatrix.

    Parameters:
        word (Str): The matrix to compute power range for.

    Returns:
        tuple: A tuple (max_positive_power, max_negative_power) representing the largest 
               positive and most negative powers of x across all matrix entries.
    """
    matrix = word_to_matrix(dictionary,word)
    return largest_power_range(matrix)

def largest_power_range(matrix):
    """
    Compute the largest positive and negative powers of x for all entries in a LaurentMatrix.

    Parameters:
        matrix (LaurentMatrix): The matrix to compute power range for.

    Returns:
        int: max(abs(power))
    """
    m = 0
    for row in matrix.matrix:
        for entry in row:
            m = max(m, max(abs(entry.min_power), abs(entry.min_power + len(entry.coefficients) - 1)))
    return m

def matrix_coefficient_sum(matrix):
    """
    Compute the sum of absolute values of all coefficients in a Laurent matrix.
    
    Parameters:
        matrix (LaurentMatrix): The Laurent matrix to analyze
        
    Returns:
        int or float: The sum of absolute values of all coefficients
    """
    total_sum = 0
    
    # Iterate through each entry in the 3x3 matrix
    for i in range(3):
        for j in range(3):
            # Get the Laurent polynomial at this position
            laurent_poly = matrix.matrix[i, j]
            
            # For each coefficient in the polynomial, add its absolute value to the sum
            for coef in laurent_poly.coefficients:
                # If in modulo context, treat the residue as an integer
                if laurent_poly.modulo is not None:
                    # Ensure the coefficient is in the range [0, modulo-1]
                    normalized_coef = coef % laurent_poly.modulo
                    # Take the smaller of the coefficient or (modulo - coefficient)
                    # to find the "absolute value" in modular arithmetic
                    abs_value = min(normalized_coef, laurent_poly.modulo - normalized_coef)
                else:
                    # Standard absolute value for non-modulo case
                    abs_value = abs(coef)
                
                total_sum += abs_value
    
    return total_sum

def process_batch(args):
    """Process a batch of entries in one process"""
    dict_ref, symbols, batch_entries, t = args
    
    if t == 0:
        return batch_entries
        
    results = {}
    for key, value in batch_entries.items():
        s = symbols.copy()
        try:
            s.remove(key[-1].swapcase())
        except (IndexError, ValueError):
            # Handle empty key or key without valid swapcase
            continue
            
        for i in s:
            results[key+i] = value * dict_ref[i]
    return results

def extend_in_all_ways_p(dict_ref, entries, t):
    """
    Extend entries by multiplying with dictionary values, dividing work into
    equal parts processed in parallel.
    """
    if t == 0:
        return entries
        
    symbols = ["A", "B", "a", "b"]
    

    num_processes = multiprocessing.cpu_count()

    # For very small entry sets or single CPU, just process sequentially
    if num_processes <= 1 or len(entries) < num_processes:
        results = {}
        for key, value in entries.items():
            s = symbols.copy()
            try:
                s.remove(key[-1].swapcase())
            except (IndexError, ValueError):
                continue
                
            for i in s:
                results[key+i] = value * dict_ref[i]
        
        return extend_in_all_ways(dict_ref, results, t-1)
    
    # Divide entries into approximately equal batches
    entries_items = list(entries.items())
    batch_size = max(1, len(entries_items) // num_processes)
    batches = []
    
    for i in range(0, len(entries_items), batch_size):
        # Take a slice of the entries
        batch = dict(entries_items[i:i+batch_size])
        entries_items[i:i+batch_size] = [None]*batch_size
        batches.append(batch)
    
    # If we didn't get enough batches, add empty ones
    while len(batches) < num_processes:
        batches.append({})
    
    # Prepare arguments for each process
    process_args = []
    for i in range(len(batches)):
        process_args.append((dict_ref, symbols, batches[i], 1))
        batches[i] = None
    
    # Process each batch in parallel
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            batch_results = pool.map(process_batch, process_args)
            
        # Combine results from all batches
        combined_results = {}
        for result_dict in batch_results:
            combined_results.update(result_dict)
            
        # Continue extending with the combined results
        return extend_in_all_ways_p(dict_ref, combined_results, t-1)
        
    except Exception as e:
        print(f"Parallel processing failed: {e}. Falling back to sequential.")
        # Fall back to sequential processing if parallel fails
        results = {}
        for key, value in entries.items():
            s = symbols.copy()
            try:
                s.remove(key[-1].swapcase())
            except (IndexError, ValueError):
                continue
                
            for i in s:
                results[key+i] = value * dict_ref[i]
                
        return extend_in_all_ways(dict_ref, results, t-1)

def extend_in_all_ways(dict,entries,t):
    if(t == 0):
        return entries
    symbols = ["A","B","a","b"]
    results = {}
    for key, value in entries.items():
        s = symbols.copy()
        s.remove(key[-1].swapcase())
        for i in s:
            results[key+i] = value*dict[i]
    return extend_in_all_ways(dict,results,t-1)

def process_word_batch(args):
    """Process a batch of entries in one process"""
    symbols, batch_entries= args
    
    results = {}
    for key, _ in batch_entries.items():
        s = symbols.copy()
        s.remove(key[-1].swapcase())         
        for i in s:
            results[key+i] = None
    return results
            
def extend_word_in_all_ways_p(entries,t):
    """
    Extend entries by multiplying with dictionary values, dividing work into
    equal parts processed in parallel.
    """
    if t == 0:
        return entries
        
    symbols = ["A", "B", "a", "b"]
    

    num_processes = multiprocessing.cpu_count()


    
    # Divide entries into approximately equal batches
    entries_items = list(entries.items())
    batch_size = max(1, len(entries_items) // num_processes)
    batches = []
    
    for i in range(0, len(entries_items), batch_size):
        # Take a slice of the entries
        batch = dict(entries_items[i:i+batch_size])
        batches.append(batch)
    
    # If we didn't get enough batches, add empty ones
    while len(batches) < num_processes:
        batches.append({})
    
    # Prepare arguments for each process
    process_args = [(symbols, batch) for batch in batches]
    # Process each batch in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        batch_results = pool.map(process_word_batch, process_args)
        
    # Combine results from all batches
    combined_results = {}
    for result_dict in batch_results:
        combined_results.update(result_dict)
        
    # Continue extending with the combined results
    return extend_word_in_all_ways_p(combined_results, t-1)

def calculate_products(dict,max_length):
    results = dict
    for i in tqdm(range(max_length)):

        results = results | extend_in_all_ways_p(dict,results,1)

    return results

def get_invariant_picture(results,invariant):
    degrees = {}
    max_degree = 0
    for key,value in results.items():
        if invariant(value) > max_degree:
            max_degree = invariant(value)
    for i in range(max_degree+1):
        degrees[i] = 0
    for key,value in results.items():
        degrees[invariant(value)] += 1
    return degrees

#A: ,BBB: , (3 - n_1,4 - n_2,7 - n_3,10,...) [1,0.5],0.1
def tiered_sampling(results, tier_percentages = [0], remaining_percentage = 0,min_num = 0,max_num = 1000000000, invariant = largest_power_range, trace = False):
    """
    Sample from different tiers of results based on specified percentages.
    
    Args:
        results: Dictionary with keys and their associated values to evaluate
        tier_percentages: List of percentages for each tier, starting from best tier
        remaining_percentage: Percentage to sample from items not in any tier
        trace: whether or not incude additional statistics
    
    Returns:
        Dictionary containing the sampled items from each tier
    """
    if not results or not tier_percentages:
        return {}
    
    # Calculate all invariants once
    invariants = {key: invariant(value) for key, value in results.items()}

    # Sort keys by their invariant values (ascending)
    grouped = defaultdict(dict)
    for key, value in invariants.items():
        grouped[value][key] = results[key]
        del results[key]
    # Calculate tier boundaries

    sorted_elements = [grouped[key] for key in sorted(grouped)]

    #for memory efficiency
    total_size = 0
    for i in range(len(sorted_elements)):
        total_size += i
        if total_size > max_num and i+1<=len(sorted_elements):
            for j in range(i+1,len(sorted_elements)):
                sorted_elements[j] = None


    final = {}
    num_tiers = len(tier_percentages)
    i = 0
    for i in range(len(sorted_elements)):
        length = len(final)
        if i == num_tiers:
            break
        elif length + len(sorted_elements[i]) >= max_num:
            final.update(dict(random.sample(list(sorted_elements[i].items()), int(max_num-length))))
            return final
        
        elif tier_percentages[i] == 0:
            continue
        elif tier_percentages[i] == 1:
            final.update(sorted_elements[i])
            sorted_elements[i] = dict()
        else:
            add_mat = dict(random.sample(list(sorted_elements[i].items()), int(tier_percentages[i]*len(sorted_elements[i]))))
            final.update(add_mat)
            sorted_elements[i] = {k: v for k, v in sorted_elements[i].items() if k not in add_mat.keys()}
    

    i = 0
    while length < min_num and i != len(sorted_elements):
        if len(sorted_elements[i]) >= min_num - length:
            new_mat = dict(random.sample(list(sorted_elements[i].items()), min_num - length))
            final.update(new_mat)
            sorted_elements[i] = dict()
            length = len(final)
        else:
            final.update(sorted_elements[i].items())
            sorted_elements[i] = dict()
            length = len(final)
        i+= 1



    return final

def compute_inv_for_batch(batch_data):
    """Compute invariants for a batch of items
    
    Args:
        batch_data: Tuple of (dict_items, invariant_function)
    
    Returns:
        Dictionary mapping keys to their computed invariant values
    """
    dict, items, invariant_function = batch_data
    result = {}
    for key, _ in items:
        result[key] = invariant_function(word_to_matrix(dict,key))
        if invariant_function(word_to_matrix(dict,key)) == 0:
            continue
    return result

def tiered_word_sampling_p(matrices,results, tier_percentages=[0], remaining_percentage=0, 
                          min_num=0, max_num=1000000, invariant=largest_power_range):
    """
    Sample from different tiers of results based on specified percentages.
    Uses parallel processing to compute invariants in batches.
    
    Args:
        results: Dictionary with keys and their associated values to evaluate
        tier_percentages: List of percentages for each tier, starting from best tier
        remaining_percentage: Percentage to sample from items not in any tier
        min_num: Minimum number of results to return
        max_num: Maximum number of results to return
        invariant: Function to compute the invariant for each value
    
    Returns:
        Dictionary containing the sampled items from each tier
    """
    if not results or not tier_percentages or invariant is None:
        return {}
    

    num_processes = multiprocessing.cpu_count()
    
    # Convert results to list of items
    items = list(results.items())
    

    # Split items into approximately equal batches
    batch_size = max(1, len(items) // num_processes)
    batches = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:min(i+batch_size, len(items))]
        batches.append((matrices,batch, invariant))
    

    with multiprocessing.Pool(processes=num_processes) as pool:
        batch_results = pool.map(compute_inv_for_batch, batches)
    # Combine results
    invariants = {}
    for batch_result in batch_results:
        invariants.update(batch_result)

    
    grouped = defaultdict(dict)
    for key, value in invariants.items():
        grouped[value][key] = results[key]
    # Calculate tier boundaries

    sorted_elements = [grouped[key] for key in sorted(grouped)]
    sort_len = []

    final = {}
    num_tiers = len(tier_percentages)
    for i in range(len(sorted_elements)):
        if i >= num_tiers:
            final.update(dict(random.sample(list(sorted_elements[i].items()), int(remaining_percentage*len(sorted_elements[i])))))
        elif tier_percentages[i] == 0:
            continue
        elif len(final) >= max_num:
            return final
        elif tier_percentages[i] == 1:
            final.update(sorted_elements[i])
        else:
            final.update(dict(random.sample(list(sorted_elements[i].items()), int(tier_percentages[i]*len(sorted_elements[i])))))
    i = 0
    while len(final) < min_num and i != len(sorted_elements):
        if len(sorted_elements[i])>=min_num:
            final.update(dict(random.sample(list(sorted_elements[i].items()), min_num)))
        else:
            final.update(sorted_elements[i].items())
        i+= 1
    return final


def min_invariant_in_array(results,invariant = largest_power_range,cutoff = 0):
    dict = []
    min = 1000000000
    for key,value in results.items():
        if invariant(value) <= min:
            min = invariant(value)
            if min == cutoff:
                dict.append(key)
                
    return min,len(dict),dict



def euclidean_norm(matrix):

    """
    Compute the Euclidean (Frobenius) norm of a matrix.
    
    The Euclidean norm is the square root of the sum of the squares of all elements.
    
    Parameters:
    matrix (numpy.ndarray): Input matrix (3x3)
    
    Returns:
    float: The Euclidean norm of the matrix
    """

    norm_value = np.linalg.norm(matrix, 'fro')

    return norm_value

def generate_words(length):
    def build(word, prev, remaining):
        if remaining == 0:
            
            return
        for g in ['A', 'a', 'B', 'b']:
            if (prev, g) in [('A', 'a'), ('a', 'A'), ('B', 'b'), ('b', 'B')]:
                continue
            build(word + g, g, remaining - 1)

    build('', '', length)

