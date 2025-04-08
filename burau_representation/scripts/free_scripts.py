import burau_enchanced as lmp
from itertools import product
from collections import defaultdict
import numpy as np 
import json
from tqdm import tqdm
import random
import multiprocessing
from functools import partial

symbol_to_matrix = {
    "A": lmp.A,
    "B": lmp.B,
    "a": lmp.a,
    "b": lmp.b
}
symbol_to_matrix3 = {
    "A": lmp.A.convert_to_modulo(3),
    "B": lmp.B.convert_to_modulo(3),
    "a": lmp.a.convert_to_modulo(3),
    "b": lmp.b.convert_to_modulo(3)
}

# Inverse symbol pairs
inverses = {"A": "a", "a": "A", "B": "b", "b": "B"}



def largest_power_range(matrix):
    """
    Compute the largest positive and negative powers of x for all entries in a LaurentMatrix.

    Parameters:
        matrix (LaurentMatrix): The matrix to compute power range for.

    Returns:
        tuple: A tuple (max_positive_power, max_negative_power) representing the largest 
               positive and most negative powers of x across all matrix entries.
    """
    m = 0
    for row in matrix.matrix:
        for entry in row:
            m = max(m, max(abs(entry.min_power),abs(entry.min_power + len(entry.coefficients) - 1)))
    return m


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
        batches.append(batch)
    
    # If we didn't get enough batches, add empty ones
    while len(batches) < num_processes:
        batches.append({})
    
    # Prepare arguments for each process
    process_args = [(dict_ref, symbols, batch, 1) for batch in batches]
    
    # Process each batch in parallel
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            batch_results = pool.map(process_batch, process_args)
            
        # Combine results from all batches
        combined_results = {}
        for result_dict in batch_results:
            combined_results.update(result_dict)
            
        # Continue extending with the combined results
        return extend_in_all_ways(dict_ref, combined_results, t-1)
        
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


def calculate_products(dict,max_length):
    results = dict
    for i in tqdm(range(max_length)):

        results = results | extend_in_all_ways_p(dict,results,1)

    return results



def extend_to_file_iteratively(dict,file_name):
    symbols = ["A","B","a","b"]
    with open(file_name, "w") as f:
        f.write("{\n")  # Start the JSON object
        first_entry = True
        for key, value in dict.items():
            s = symbols.copy()
            s.remove(key[-1].swapcase())
            for i in s:
                if not first_entry:
                    f.write(",\n")  
                print(key+i)
                json.dump(key+i, f)
                f.write(": ")
                a = (value*symbol_to_matrix[i]).to_nested_list()
                json.dump(a,f)
                first_entry = False
        
        f.write("\n}")  # Close the JSON object


def generate_reduced_word(length, alphabet):
    """
    Generate a random reduced word of a given length.
    
    Parameters:
        length (int): The desired length of the reduced word.
        alphabet (list of str): The alphabet of the group, including inverses. 
                                Example: ["a", "A", "b", "B", "c", "C"]
                                
    Returns:
        list of str: A reduced word represented as a list of letters.
    """
    if length <= 0:
        return []

    word = []
    while len(word) < length:
        # Randomly choose a letter from the alphabet
        next_letter = random.choice(alphabet)
        
        # Avoid consecutive inverses
        if word and word[-1].swapcase() == next_letter:
            continue
        
        word.append(next_letter)

    return word



def get_degree_picture(results,n):
    degrees = {}
    for i in range(n+1):
        degrees[i] = 0
    for key,value in results.items():
        degrees[largest_power_range(value)] += 1
    print(degrees)



def cut_by_degree(results,n):
    dict = {}
    min = 1000000000
    for key,value in results.items():
        if largest_power_range(value) <= min:
            min = largest_power_range(value)
    for key,value in results.items():
        if largest_power_range(value) <= n+min:
            dict[key] = value
    return dict



def tiered_sampling(results, tier_percentages, remaining_percentage,min_num,max_num):
    """
    Sample from different tiers of results based on specified percentages.
    
    Args:
        results: Dictionary with keys and their associated values to evaluate
        tier_percentages: List of percentages for each tier, starting from best tier
        remaining_percentage: Percentage to sample from items not in any tier
    
    Returns:
        Dictionary containing the sampled items from each tier
    """
    if not results or not tier_percentages:
        return {}
    
    # Calculate all invariants once
    invariants = {key: largest_power_range(value) for key, value in results.items()}
    # Sort keys by their invariant values (ascending)
    grouped = defaultdict(dict)
    for key, value in invariants.items():
        grouped[value][key] = results[key]
    # Calculate tier boundaries

    sorted_elements = [grouped[key] for key in sorted(grouped)]

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
    
    
    



def cut_by_invariant(results,invariant,n):
    dict = {}
    min = 1000000000
    for key,value in results.items():
        if largest_power_range(value) <= min:
            min = largest_power_range(value)
    for key,value in results.items():
        if largest_power_range(value) <= n+min:
            dict[key] = value
    return dict

def cut_by_invariant(results,invariant,n):
    dict = {}
    min = 1000000000
    for key,value in results.items():
        if largest_power_range(value) <= min:
            min = largest_power_range(value)
    for key,value in results.items():
        if largest_power_range(value) <= n+min:
            dict[key] = value
    return dict

def min_degree_in_array(results):
    dict = []
    min = 1000000000
    for key,value in results.items():
        if largest_power_range(value) <= min:
            min = largest_power_range(value)
            if min == 0:
                dict.append(key)
                
    return min,len(dict),dict

def word_to_matrix(dict,word):
    result  = dict[word[0]]
    for i in range(1,len(word)):
        result *= dict[word[i]]
    return result

def compute_laurent_matrix_invariants(matrix):
    """
    Compute various invariants of a Laurent polynomial matrix.
    
    Parameters:
        matrix (LaurentMatrix): The matrix to analyze
        
    Returns:
        dict: A dictionary containing various invariants of the matrix:
            - max_degree: Maximum degree of any polynomial in the matrix
            - min_degree: Minimum degree of any polynomial in the matrix
            - degree_span: Difference between max and min degree
            - max_abs_degree: Maximum of the absolute values of min and max degrees
            - total_terms: Total number of non-zero terms across all polynomials
            - sparsity: Ratio of zero entries to total entries
            - entry_degrees: 2D array showing the degree span of each entry
            - degree_distribution: Dictionary counting polynomials by degree
    """
    
    rows, cols = matrix.matrix.shape
    
    # Initialize tracking variables
    max_degree = float('-inf')
    min_degree = float('inf')
    total_terms = 0
    dense_entries = 0
    entry_degrees = [[None for _ in range(cols)] for _ in range(rows)]
    degree_distribution = {}
    
    # Iterate through matrix entries
    for i in range(rows):
        for j in range(cols):
            poly = matrix.matrix[i, j]
            
            # Skip if polynomial has no terms
            if len(poly.coefficients) == 0:
                continue
                
            # Find indices of non-zero coefficients
            non_zero_indices = np.nonzero(poly.coefficients)[0]
            if len(non_zero_indices) == 0:
                continue
            
            # Calculate degree range for this polynomial
            poly_min_power = poly.min_power
            poly_max_power = poly.min_power + len(poly.coefficients) - 1
            
            # Find min and max powers considering only non-zero coefficients
            if len(non_zero_indices) > 0:
                actual_min_power = poly.min_power + non_zero_indices[0]
                actual_max_power = poly.min_power + non_zero_indices[-1]
                
                # Update global min and max
                min_degree = min(min_degree, actual_min_power)
                max_degree = max(max_degree, actual_max_power)
                
                # Store degree info for this entry
                entry_degrees[i][j] = (actual_min_power, actual_max_power)
                
                # Count total non-zero terms
                total_terms += len(non_zero_indices)
                
                # Update degree distribution
                degree_span = actual_max_power - actual_min_power
                if degree_span in degree_distribution:
                    degree_distribution[degree_span] += 1
                else:
                    degree_distribution[degree_span] = 1
    
    # If no non-zero entries were found
    if max_degree == float('-inf'):
        max_degree = 0
        min_degree = 0
    
    # Calculate matrix-wide metrics
    degree_span = max_degree - min_degree
    
    # New invariant: maximum of the absolute values of min and max degrees
    max_abs_degree = max(abs(min_degree), abs(max_degree))
    
    sparsity = 1 - (dense_entries / (rows * cols))
    
    return {
        "max_abs_degree": max_abs_degree,
        "total_terms": total_terms,
        "sparsity": sparsity,
        "entry_degrees": entry_degrees,
        "degree_distribution": degree_distribution
    }

def get_matrices_within_threshold(matrices_dict, k=1, max_count=100000000000, custom_goals=None):

    """
    For each invariant, find matrices with values within k of the best value and return their union.
    
    Parameters:
        matrices_dict (dict): Dictionary mapping names to LaurentMatrix objects
        k (float/int): Threshold difference from the best value
        max_count (int): Maximum number of matrices to include per invariant
        custom_goals (dict): Dictionary mapping invariant names to 'min' or 'max'
                            If None, uses default optimization goals
        
    Returns:
        dict: Dictionary mapping matrix names to the original matrices,
              containing only those that were within threshold k of the best for at least one invariant
    """
    if not matrices_dict:
        return {}
    
    # Default optimization goals for common invariants
    default_goals = {
        "max_degree": "min",         # Lower max degree is better
        "min_degree": "max",         # Higher min degree is better
        "degree_span": "min",        # Lower degree span is better
        "max_abs_degree": "min",     # Lower max absolute degree is better
        "total_terms": "min",        # Fewer terms is better (more compact)
    }
    
    goals = custom_goals if custom_goals is not None else default_goals
    
    # Compute invariants for all matrices
    all_invariants = {}
    for name, matrix in matrices_dict.items():
        all_invariants[name] = compute_laurent_matrix_invariants(matrix)
    
    # Set to track all matrices that are within threshold of best for at least one invariant
    qualifying_matrices = set()
    
    # For each invariant, find matrices within threshold k of the best value
    for invariant, goal in goals.items():
        # Skip invariants that don't exist in the computed data
        if invariant not in all_invariants[list(all_invariants.keys())[0]]:
            continue
            
        # Determine if we're maximizing or minimizing
        is_maximize = (goal.lower() == "max")
        
        # Find the best value for this invariant
        if is_maximize:
            best_value = max(all_invariants[name][invariant] for name in matrices_dict)
            threshold = best_value - k
            
            # Get matrices within threshold, sorted by value
            qualifying = [(name, all_invariants[name][invariant]) 
                         for name in matrices_dict 
                         if all_invariants[name][invariant] >= threshold]
                         
            # Sort by value in descending order
            qualifying.sort(key=lambda x: x[1], reverse=True)
            
        else:  # Minimizing
            best_value = min(all_invariants[name][invariant] for name in matrices_dict)
            threshold = best_value + k
            
            # Get matrices within threshold, sorted by value
            qualifying = [(name, all_invariants[name][invariant]) 
                         for name in matrices_dict 
                         if all_invariants[name][invariant] <= threshold]
                         
            # Sort by value in ascending order
            qualifying.sort(key=lambda x: x[1])
        
        # Limit number of matrices to max_count
        qualifying = qualifying[:max_count]
        
        # Add these matrices to our qualifying set
        qualifying_matrices.update(name for name, _ in qualifying)
    
    # Create the result dictionary with only the qualifying matrices
    result = {name: matrices_dict[name] for name in qualifying_matrices}
    
    return result

print(word_to_matrix(symbol_to_matrix,"aBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaBABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBAB"))
