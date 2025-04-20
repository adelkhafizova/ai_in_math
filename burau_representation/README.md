# Description of files:
- **scripts**, contains main logic 
  - burau_enchanced.py, contains definitions of classes LaurentPolynomial and LaurentMatrix, and definitions of matrices A,B from the paper https://arxiv.org/pdf/1904.11730.
  
  - free_scripts.py, contains functions for working with free words over two letters and words in matrices A,B.

- **experiments**, contains code and logs for experiments
  - server_computations.py, search for a word in A,B that gives an identity matrix modulo 2, but with more computational resourses. Found words of minimal length 32.
  
  - mod3.py, search for a word in A,B that gives an identity matrix modulo 3. Found words of minimal length 338.
  
  - mod5.py, search for a word in A,B that gives an identity matrix modulo 5. Didn't find a word
  
  - find_mod6.py, search for a word in A,B that gives an identity matrix modulo 6 by using known words modulo 2 and 3. Didn't find a word.
  
  - find_all_good_words_15_2.py, search over all reduced words of length 15 and the value of largest_power_range less that 5.

  - stress_test_matrices.py, test whether some functions defined in free_scripts.py and burau_enchanced.py utilize all cpu in nebius vm.

