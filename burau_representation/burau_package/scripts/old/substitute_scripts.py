import ai_in_math.burau_representation.scripts.old.laurent_modulo_p as b
from itertools import product
import numpy as np 
import math
import json
symbol_to_matrix = {
    "A": b.A,
    "B": b.B,
    "a": b.a,
    "b": b.b
}

# Inverse symbol pairs
inverses = {"A": "a", "a": "A", "B": "b", "b": "B"}

def is_reduced_word(word):
    """
    Check if a word is reduced (no consecutive inverse pairs).
    """
    for i in range(len(word) - 1):
        if word[i + 1] == inverses.get(word[i]):
            return False
    return True


def load_specific_element_incrementally(filename, key):
    """
    Load a specific element by its key from a large JSON file incrementally.
    
    Parameters:
        filename (str): Path to the JSON file.
        key (str): The key of the element to load.
    
    Returns:
        Any: The value associated with the specified key, or None if not found.
    """
    with open(filename, "r") as f:
        # Read the file line by line for large dictionaries
        for line in f:
            # Skip lines that are not part of key-value pairs
            line = line.strip()
            if line.startswith("{") or line.startswith("}"):
                continue  # Skip the dictionary braces
            
            # Ensure the line ends properly
            if line.endswith(","):
                line = line[:-1]  # Remove trailing comma
            
            # Parse the line as a key-value pair
            try:
                entry = json.loads(f"{{{line}}}")  # Wrap it to parse as a dictionary
                if key in entry:
                    return entry[key]  # Return the value if the key matches
            except json.JSONDecodeError:
                continue  # Skip malformed lines
            
    return None  # Return None if the key was not found


def search_key_in_json(file_path, target_key):
    """
    Search for a specific key in a large JSON Lines file.
    
    Parameters:
        file_path (str): Path to the JSON Lines file.
        target_key (str): The key to search for.
    
    Returns:
        dict or None: The value corresponding to the target_key, or None if not found.
    """
    with open(file_path, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            if target_key in record:
                return record[target_key]
    return None


def extend_in_all_ways(entries,t):
    if(t == 0):
        return entries
    symbols = ["A","B","a","b"]
    results = {}
    for key, value in entries.items():
        s = symbols.copy()
        s.remove(key[-1].swapcase())
        for i in s:
            results[key+i] = value*symbol_to_matrix[i]
    return extend_in_all_ways(results,t-1)

def calculate_to_n(n):
    init = symbol_to_matrix
    final = init
    for i in range(n):
        print(i)
        init = extend_in_all_ways(init,1)
        final = init | final
    return final


def calculate_products(max_length,file_name):
    results = symbol_to_matrix
    final = results
    with open(file_name, "a") as f:
        f.write("[\n")  # Start the JSON array
        json.dump(results.items(), f)
        f.write(",\n")
        for i in range(max_length):
            print(i)
            results = extend_in_all_ways(results,1)
            json.dump(results, f)
            if(i!=max_length-1):
                f.write(",\n")
            final = results | final
        f.write("\n]")
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

def is_identity(A):
    id = np.eye(A.shape[0])
    return np.allclose(A,id)
