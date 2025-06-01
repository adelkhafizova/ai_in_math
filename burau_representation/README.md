# Problem statement and description of current approach
The reduced Burau repsesentation of dimension $n$ is a homeomorphism $\rho_{n} :B_{n} \rightarrow GL_{n-1}(Z[t,t^{-1}])$ defined on a standart generators of a braid group  as follows:

<img width="483" alt="image" src="https://github.com/user-attachments/assets/dc5f9785-bab6-41e4-868f-2743da5138af" />

For $n=3$ it is known to be faithful, for $n\geq5$ it is known to be not faithful. But determining it's faithfulness for $n = 4$ is an open problem.

In https://arxiv.org/pdf/1904.11730 it is described that faithfulness for $n=4$ is equivalent to some two matrices $A,B \in GL_{3}(Z[t,t^{-1}])$ generating a free group, the matrices are:

<img width="492" alt="image" src="https://github.com/user-attachments/assets/c26972c3-a431-486d-b113-b65c4a0582ec" />

We trying to disprove the conjecture by finding reduced word in $A,B,A^{-1} =: a,B^{-1} =: b$ that is equal to $I_{3}$.

The question may be simplified by considering coefficients of $A,B$ modulo some number $p$. Using simple deterministic search that will be described later we found:

- For $p = 2$, word of minimal length $32$ that gives identity.
- For $p = 3$, word of minimal length $338$ that gives identity.
- For $p = 5$, did not find anything.

This is not surprising since the fact that reduced Burau representation is not faithful for $p= 2,3$ is known result (see https://arxiv.org/pdf/1904.11730), but question for $p=5$ is remains open.

### Description of search modulo p

Let's define $W(p)$ being space of all matrices that corrseponding to all reduced words in $A,B$, with coefficients modulo $p$. Furher we identify reduced words with matrices to which it correspond.

So we want to find $w \in W(p)$, such that $w = I_{3}$. 

Let's add other nontation $W_{i}(p)$ for reduced words of length $i$. Ideally, we would like iteratively go through all $W_i(p)$ and check whether it is identity, but the search space becoming too big for brute force even for relatively small $i$. The idea is for each $i$ to check only some subset $S_i \subset W_{i}(p)$.

Given some subset $S \subset W_{i}(p)$, let's define $f(S) \subset W_{i+1}(p)$ as all possible extension of words of $S$ by one letter. 

For example if $S = [AA,AB]$, then $f(S) = [AAA,AAB,AAb,ABA,ABB,ABa]$.

Another important procees for our approach is how to given $S \subset W_{i}(p)$ obtain some "good" subset $S'$ of $S$, which will be small relative to $S$. 

The approach that we took is to compute some invariant for all matrices in $S$, that will measure how close is matrix to identity, and then take subset of $S$ consisting of $k$ words with smallest values of this invariant, for some natural $k$.

Let's describe invariant that we looked at, for laurent polynomial over any field $P$, define $d(P)$ being maximal aboute value of degrees of all it monomials, for example $d(t^{-100}+t) = 100,d(3t^{-1}+t^2) = 2$. For matrix 
$M \in GL_{3}(Z_{p}[t,t^{-1}])$ define $d(M) = max(d(m_{ij}))$. We see that $d(A) = d(B) = 1$.

We now ready to give a pseudocode for an algorithm:

```text
ALGORITHM MatrixSearch:
    
    n ← 10
    S_n = W_n //compute whole W_n for some base n

    WHILE TRUE:
        sort S_i by the value of d(w), for each w in S_i
        take S being first k words
        set S_i+1 = f(S)
        if S_i+1 has word w which gives identity matrix:
          return w
```

The plan for now is to try to implement some RL algorithms to improve the performance for $p=2,3$ and to find the word for $p = 5$.

### RL


# Project Structure
- **scripts**, contains main logic, documentation contained in readme file inside a folder.
  - burau_enchanced.py, contains definitions of classes LaurentPolynomial and LaurentMatrix, and definitions of matrices A,B from the paper https://arxiv.org/pdf/1904.11730.
  
  - free_scripts.py, contains functions for working with free words over two letters and words in matrices A,B.

- **experiments**, contains code and logs for experiments. Data obtained during experiments is stored locally (either on nebius virtual machine or on my laptop, in case you want to obtain it, you can contact me via boris2107g@gmail.com) , in some cases when it is really small it may be stored here. 
  - server_computations.py, search for a word in A,B that gives an identity matrix modulo 2, but with more computational resourses. Found words of minimal length 32.
  
  - mod3.py, search for a word in A,B that gives an identity matrix modulo 3. Found words of minimal length 338.
  
  - mod5.py, search for a word in A,B that gives an identity matrix modulo 5. Didn't find a word
  
  - find_mod6.py, search for a word in A,B that gives an identity matrix modulo 6 by using known words modulo 2 and 3. Didn't find a word.
  
  - find_all_good_words_15_2.py, search over all reduced words of length 15 and the value of largest_power_range less that 5.

  - stress_test_matrices.py, test whether some functions defined in free_scripts.py and burau_enchanced.py utilize all cpu in nebius vm.

- **RL**, contains everything related with RL
  - burauDQNmod2, implementation of simple DQN for finding words modulo $2$, the best model so far is able to find only words of length $34$ (the best done deterministically is  $32$).

- **old** contains code that is irrelevant for current approach, but may be needed in future.

