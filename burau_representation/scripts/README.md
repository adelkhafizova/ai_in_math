# Description of some selected functions. Among all functions _p at the end stands for parralel implementation of a function.
- invariants for LaurentMatrix defined in free_scripts.py. They are functions that receive LaurentMatrix and return real number:
    -largest_power_range, maximal absolute value of power of all monomials in all entries of matrices.
    -matrix_coefficient_sum, sum of absolute values of all coefficients of all polynomials in all entries of a matrix (for example for identity matrix equal to 3). For modulo version it just sums residues as if they were integers, when they are considered as 0,1,...,n-1.
    -euclidean_norm, for matrix M, returns Euclidean (Froblenius) norm of a M-I, where I is an identity matrix.
### `extend_in_all_ways(dict_ref,entries,t)`

**Purpose:**  
Extend all words in entries t times (recursively) by using matrices A,B,a,b, where they are passed through dictionary dict_ref. 

**Parameters:**  
- `dict_ref` (dict(LaurentMatrix)): Dictionary that has four elements with keys "A","B","a","b", and values associated to the key being LaurentMatrix that correspond to the letter.

- `entries` (dict(LaurentMatrix)): Dictionary contains keys being some reduced words in A,B,a,b and values being LaurentMatrix that correspond to the key.

- `t` (int): Number of times that we are extending the matrices in entires.

**Returns:**  
- `dict(LaurentMatrix)`: Dictionary that is obtained by extending entries.

**Example Usage:**
```python
dict_ref = {
        "A": A,
        "B": B,
        "a": a,
        "b": b
    }
entries = {
        "A": A,
        "BbaB": BbaB,

}
result = extend_in_all_ways(dict_ref,entries,1)
print(result)  # Output: {"AA": AA, "AB": AB, "Ab": Ab, "BbaBA" : BbaBA, "BbaBa" : BbaBa, "BbaBB" : BbaBB}
```
### `tiered_sampling(results,tier_percentages,remaining_percentage,min_num,max_num,invariant)`

**Purpose:**  
Returns subdictionary of `results` with "good" values of `invariant`, this subdictionary is called `final`. 

It divides dictionary to subdictionaries with the same value of invariant, and sorts these subdictionaries from smallest to biggest value, the list of subdictionaries is called `sorted_elements`. 

Then the code procced as follows, for each `i` it adds `tier_percentages[i]*len(sorted_elements[i])` random matrices from `sorted_elements[i]` to `final`, so if for example `tier_percentages[0] = 1`, then it adds to `final` all matrices for which the value of `invariant` is smallest. 

Then it adds to `final` `remaining_precentage*(len(results)-len(final))` random matrices from what remained. The idea is to add some matrices with maybe big value of invariant that migth give smaller value later.

Then if size of `final` smaller then `min_num` it adds matrices so that the size will become bigger.

At each step the code checks if size of `final` is bigger than `max_num`, and if so, returns `final`. 


**Parameters:**  
- `results` (dict(LaurentMatrix)): Dictionary contains keys being some reduced words in A,B,a,b and values being LaurentMatrix that correspond to the key.

- `tier_percentages` (list(float)): Dictionary contains numbers from 0 to 1.

- `remaining_percentage` (float): Number from 0 to 1

- `min_num` (int)

- `max_num` (int)

- `invariant` function that receives `LaurentMatrix` and returns an `int`.

**Returns:**  
- `final`: Subdictionary of `results`.




