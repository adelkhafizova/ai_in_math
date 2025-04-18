from collections import deque
import itertools
from sage.all import *
from pysat.formula import *
from pysat.solvers import *

K = GF(2)
gens = [matrix(QQ, 4, [1, 0, 0, 1, 0, -1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 1]), matrix(QQ, 4, [-1, 0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 1, 0, 0, 0, 1])]
P = MatrixGroup(gens)

N = 5
symbols = [P.one(), P.gen(0), P.gen(0).inverse(), P.gen(1), P.gen(1).inverse()]
B = [prod(word) for word in itertools.product(symbols, repeat=N)]
B = list(set(B))
print("Length of ball of radius ", N, ": ", len(B))
# print(B[0])

# Keys are the product, values are lists of pairs realizing the product
product_table = dict()
for i,j in itertools.product(range(len(B)), repeat=2):
    a, b = B[i], B[j]
    val = a*b
    if val in product_table:
        product_table[val].append((i, j))
    else:
        product_table[val] = [(i,j),]
print("Size of product table: ", len(product_table))

a_vars = [Atom(f"a_{i}") for i in range(len(B))]
b_vars = [Atom(f"b_{j}") for j in range(len(B))]

x_vars = dict()
cnf = CNF()

# Non triviality
formula = Equals(a_vars[0], PYSAT_TRUE)
cnf.extend([c for c in formula])

formula = Or(*list(a_vars[i] for i in range(1, len(B))))
cnf.extend([c for c in formula])

# Product equations x_g,h = a_g * b_h
for i, j in itertools.product(range(len(B_5)), repeat=2):
    x_vars[(i, j)] = Atom(f"x_{i}{j}")
    formula = Equals(x_vars[(i, j)], And(a_vars[i], b_vars[j]))
    cnf.extend([c for c in formula])

max_id = -1
for c in cnf.clauses:
    for variable in c:
        if abs(variable) > max_id:
            max_id = abs(variable)

# print(max_id)
next_id = max_id + 1

# Sum equations sum_{gh=k}(x_g,h) = delta(1,k) for each k in the product table
var_list = deque([x_vars[(i, j)] for i, j in product_table[P.one()]])
formula_list = []
while var_list:
    if len(var_list) == 1:
        formula = Equals(var_list.pop(), PYSAT_TRUE)
        formula_list.append(formula)
    elif len(var_list) > 2:
        x1 = var_list.pop()
        x2 = var_list.pop()
        aux_var = Atom(next_id)
        var_list.append(aux_var)
        next_id += 1
        formula = Equals(XOr(x1, x2), aux_var)
        formula_list.append(formula)
    elif len(var_list) == 2:
        x1 = var_list.pop()
        x2 = var_list.pop()
        formula = XOr(x1, x2)
        formula_list.append(formula)
for f in formula_list:
    cnf.extend([c for c in f])

for c in cnf.clauses:
    for variable in c:
        if abs(variable) > max_id:
            max_id = abs(variable)
next_id = max_id + 1

formula_list = []
for val in product_table:
    if not val.is_one():

        var_list = deque([x_vars[(i, j)] for i, j in product_table[val]])
        while len(var_list) > 0:
            if len(var_list) == 1:
                formula = Neg(var_list.pop())
                formula_list.append(formula)
            elif len(var_list) > 2:
                x1 = var_list.pop()
                x2 = var_list.pop()
                aux_var = Atom(next_id)
                var_list.append(aux_var)
                next_id += 1
                formula = Equals(XOr(x1, x2), aux_var)
                formula_list.append(formula)
            elif len(var_list) == 2:
                x1 = var_list.pop()
                x2 = var_list.pop()
                formula = Neg(XOr(x1, x2))
                formula_list.append(formula)

for f in formula_list:
    cnf.extend([c for c in f])

for c in cnf.clauses:
    for variable in c:
        if abs(variable) > max_id:
            max_id = abs(variable)

print("Maximum Variable ID:", max_id)

solution = []
with Minisat22(bootstrap_with=cnf.clauses) as m:
    print(m.solve())
    solution = m.get_model()
support = list(filter(lambda n: n > 0, solution))
# print(support)
# print(solution)

obj2id = Formula.export_vpool().obj2id

a_support = []
b_support = []
for i in range(len(B)):
    if obj2id[a_vars[i]] in support:
        a_support.append(i)
    if obj2id[b_vars[i]] in support:
        b_support.append(i)

with open("output.txt", "w") as f:
    f.write("A Support\n")
    f.write("-----------------\n")
    f.writelines("\n".join(map(str, a_support)))
    f.write("B Support\n")
    f.write("-----------------\n")
    f.writelines("\n".join(map(str, b_support)))
