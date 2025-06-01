from sympy.combinatorics.free_groups import free_group, vfree_group, xfree_group
from sympy.combinatorics.fp_groups import FpGroup, CosetTable, coset_enumeration_r



F, a, b = free_group("a, b")
Cox = FpGroup(F, [])
C_r = coset_enumeration_r(Cox, [a**3,b,a*b*a**-2,a**2*b*a**-1])
print(len(C_r.table))