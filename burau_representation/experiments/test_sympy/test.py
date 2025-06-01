import sympy as sp
x = sp.symbols('x')
import time
import burau_sympy as bsp
import sys
sys.path.append('../../scripts')
import burau_enchanced as be


start = time.time()

print("Small Polynomials addition test:")
sympy_p = sp.Poly(1+2*x+x**2)
classic_p = be.LaurentPolynomial([1,2,1],-1)
print("Polynomial: "+str(sympy_p))
start = time.time()
for i in range(10000):
    classic_p+classic_p
    

end = time.time()
print(f"10000 iterations for hardcoded implementation: {end - start:.6f} seconds")
start = time.time()

for i in range(10000):
    sympy_p+sympy_p

end = time.time()
print(f"10000 iterations for sympy implementation: {end - start:.6f} seconds")
print("\n")






print("Small Polynomials nultiplication test:")
print("Polynomial: "+str(sympy_p))
start = time.time()

for i in range(10000):
    classic_p*classic_p

end = time.time()
print(f"10000 iterations for hardcoded implementation: {end - start:.6f} seconds")
start = time.time()

for i in range(10000):
    sympy_p*sympy_p

end = time.time()
print(f"10000 iterations for sympy implementation: {end - start:.6f} seconds")
print("\n")








print("Small Polynomials power test:")
print("Polynomial: "+str(sympy_p))
start = time.time()

a = classic_p
for i in range(10000):
    a = a*classic_p

end = time.time()
print(f"10000 power for hardcoded implementation: {end - start:.6f} seconds")
start = time.time()

a = sympy_p
for i in range(10000):
    a = a*sympy_p

end = time.time()
print(f"10000 power for sympy implementation: {end - start:.6f} seconds")
print("\n")


