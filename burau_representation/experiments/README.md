
The pseudocode of current appoach to seach for words:

```python
"""
generate all words of length n (picked between 10 and 15) and all matrices that correspond to them (where field over which matrices are taken is also specified),
it stores them as dictionary with keys being words in "A,B,a,b" and values objects of type LaurentMatrix.
"""
current = generate_base(field,n)

"""
Iteratively do the following, replace current with extend_in_all_ways({"A":A,"B":B,"a":a,"b":b},current,1), replace current with tiered_sampling(current,parameters,invariant).
"""
while(True): #some big number of iterations
  current = extend_in_all_ways({"A":A,"B":B,"a":a,"b":b},current,1)
  current = tiered_sampling(current,parameters,invariant)

  write_good(current)#write some statistics or check whether it contains words that are identity
