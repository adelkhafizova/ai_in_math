import numpy as np
import sympy as sp
from itertools import product 
class LaurentPolynomial:
    def __init__(self, poly, min_power, modulo=None):

        self.min_power = int(min_power)
        self.poly = poly
        self.modulo = modulo





    def __mul__(self, other):
        """
        Multiplication of Laurent polynomials
        """
        return LaurentPolynomial(self.poly*other.poly,self.min_power+other.min_power,self.modulo)
        

    def __add__(self, other):
        x = self.poly.gens[0]
        m1 = self.min_power
        m2 = other.min_power
        p1 = self.poly
        p2 = other.poly
        if m1 < m2:
            p2 = p2*(x**(m2-m1))
        elif m2 < m1:
            m1 = m2
            p1 = p1*(x**(m1-m2))
            

        new_poly = p1+p2
        min_degree = new_poly.monoms()[-1][0]
        if min_degree == 0:
            return LaurentPolynomial(new_poly,m1+min_degree,self.modulo)
        else:
            new_poly = sp.Poly([new_poly.coeffs()],x,modulus = self.modulo)

            return LaurentPolynomial(new_poly,m1+min_degree,self.modulo)

    def __str__(self):
            """
            String representation of the Laurent polynomial with negative powers.
            """
            if self.poly == 0:
                result = "0"
            else:
                x = self.poly.gens[0]

                # Get the polynomial expression
                poly_expr = self.poly.as_expr()
                # Adjust the expression by shifting to negative powers

                terms = []
                for monom, coeff in zip(self.poly.monoms()[::-1], self.poly.coeffs()[::-1]):
                    terms.append(f"{coeff}x^{monom[0]+self.min_power}")

                # Join the terms with a plus sign and adjust for negative powers
                result = " + ".join(terms)
                
            # Add modulo information if applicable
            if self.modulo:
                result += f" (mod {self.modulo})"
            else:
                # Check if the domain is a finite field (e.g., GF(n)) and add modulus if so
                if isinstance(self.poly.domain, sp.FiniteField):
                    modulus = self.poly.domain.characteristic
                    result += f" (mod {modulus})"

            return result
            
    def is_zero(self):
        return self.poly == 0

    def is_one(self):
        return self.poly == 1

class LaurentMatrix:
    def __init__(self, matrix, modulo=None):
        """
        Optimized for 3x3 matrices with large Laurent polynomials.
        matrix: A 2D array where each entry is a LaurentPolynomial.
        modulo: If provided, all operations will be performed modulo this value.
        """
        self.modulo = modulo
        

        # Convert list to numpy array for faster operations
        self.matrix = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                self.matrix[i, j] = matrix[i][j]



    def __add__(self, other):
        if isinstance(other, LaurentMatrix):
            # Create a new matrix with the same modulo as self - optimized for 3x3
            result = np.empty((3, 3), dtype=object)
            
            for i in range(3):
                for j in range(3):
                    result[i, j] = self.matrix[i, j] + other.matrix[i, j]
                    
            return LaurentMatrix(result, self.modulo)
        else:
            raise ValueError("Matrix dimensions must match for addition.")

    def __mul__(self, other):
        """Optimized matrix multiplication for 3x3 matrices with large polynomials."""
        if isinstance(other, LaurentMatrix):
            # For 3x3 matrices, we can fully unroll the loops
            result = np.empty((3, 3), dtype=object)
            
            # Cache is_zero checks to avoid repeated calculations
            zero_check_self = np.empty((3, 3), dtype=bool)
            zero_check_other = np.empty((3, 3), dtype=bool)
            
            for i in range(3):
                for j in range(3):
                    zero_check_self[i, j] = self.matrix[i, j].is_zero()
                    zero_check_other[i, j] = other.matrix[i, j].is_zero()
            
            # Unrolled 3x3 matrix multiplication with zero checks
            for i in range(3):
                for j in range(3):
                    # Start with zero
                    sum_poly = LaurentPolynomial(sp.Poly(0,self.matrix[0][0].poly.gens[0]), 0, self.modulo)
                    
                    # Unroll k loop for 3x3 matrix
                    # k = 0
                    if not zero_check_self[i, 0] and not zero_check_other[0, j]:
                        sum_poly = sum_poly + (self.matrix[i, 0] * other.matrix[0, j])
                        
                    # k = 1
                    if not zero_check_self[i, 1] and not zero_check_other[1, j]:
                        sum_poly = sum_poly + (self.matrix[i, 1] * other.matrix[1, j])
                        
                    # k = 2
                    if not zero_check_self[i, 2] and not zero_check_other[2, j]:
                        sum_poly = sum_poly + (self.matrix[i, 2] * other.matrix[2, j])
                    
                    result[i, j] = sum_poly
            
            return LaurentMatrix(result, self.modulo)
        else:
            raise ValueError("Matrix dimensions must align for multiplication.")
        
    def __pow__(self, n):
        """
        Compute the nth power of a 3x3 LaurentMatrix using binary exponentiation.
        Optimized for large polynomials.
        """
        x = self.matrix[0][0].poly.gens[0]
        if not isinstance(n, int):
            raise ValueError("The exponent must be an integer.")
        if n < 0:
            raise ValueError("Negative powers are not supported yet.")
            
        # Handle special cases
        if n == 0:
            # Identity matrix - optimized creation for 3x3
            result = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        result[i, j] = LaurentPolynomial(sp.Poly(1,x), 0, self.modulo)
                    else:
                        result[i, j] = LaurentPolynomial(sp.Poly(0,x), 0, self.modulo)
            return LaurentMatrix(result, self.modulo)
        elif n == 1:
            # Return a copy of self
            result = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    result[i, j] = LaurentPolynomial(
                        self.matrix[i, j].coefficients.copy(),
                        self.matrix[i, j].min_power,
                        self.modulo
                    )
            return LaurentMatrix(result, self.modulo)
            
        # Binary exponentiation (square and multiply algorithm)
        # Create identity matrix
        result = np.empty((3, 3), dtype=object)
        for i in range(3):
                for j in range(3):
                    if i == j:
                        result[i, j] = LaurentPolynomial(sp.Poly(1,x), 0, self.modulo)
                    else:
                        result[i, j] = LaurentPolynomial(sp.Poly(0,x), 0, self.modulo)

        result_matrix = LaurentMatrix(result, self.modulo)
        power = self
        
        while n > 0:
            if n % 2 == 1:
                result_matrix = result_matrix * power
            power = power * power
            n //= 2
            
        return result_matrix

    def __str__(self):
        rows = ["[" + ", ".join(str(entry) for entry in row) + "]" for row in self.matrix]
        result = "[\n " + ",\n ".join(rows) + "\n]"
        if self.modulo is not None:
            result = f"Matrix (mod {self.modulo}):\n" + result
        return result
    
    def __call__(self, x):
        """Evaluate the 3x3 matrix at a specific value of x."""
        result = np.zeros((3, 3), dtype=float if self.modulo is None else int)
        for i in range(3):
            for j in range(3):
                result[i, j] = self.matrix[i, j](x)
                
        if self.modulo is not None:
            result = np.mod(result, self.modulo)
        return result
    
"""
    def convert_to_modulo(self, p):
        if not isinstance(p, int) or p <= 0:
            raise ValueError("Modulus must be a positive integer.")
            
        # Create a new 3x3 matrix with the modulo set to p
        new_matrix = np.empty((3, 3), dtype=object)
        
        for i in range(3):
            for j in range(3):
                new_matrix[i, j] = LaurentPolynomial(
                    self.matrix[i, j].coefficients.copy(), 
                    self.matrix[i, j].min_power, 
                    p
                )
        
        return LaurentMatrix(new_matrix, p)
"""



x = sp.symbols('x')



A = LaurentMatrix([[LaurentPolynomial(sp.Poly(0,x),0),LaurentPolynomial(sp.Poly(0,x),0),LaurentPolynomial(sp.Poly(-1,x),-1)],
                   [LaurentPolynomial(sp.Poly(0,x),0),LaurentPolynomial(sp.Poly(-1,x),1),LaurentPolynomial(sp.Poly(-1+x**2),-1)],
                   [LaurentPolynomial(sp.Poly(-1,x),0),LaurentPolynomial(sp.Poly(0,x),0),LaurentPolynomial(sp.Poly(-1+x),-1)]])


"""
B = LaurentMatrix([[LaurentPolynomial([-1],-1),1,0],
                   [0,1,0],
                   [0,1,LaurentPolynomial([-1],1)]])

a = LaurentMatrix([[LaurentPolynomial([1,-1],0),0,-1],
                   [LaurentPolynomial([1,0,-1],-1),LaurentPolynomial([-1],-1),0],
                   [LaurentPolynomial([-1],1),0,0]])

b = LaurentMatrix([[LaurentPolynomial([-1],1),LaurentPolynomial([1],1),0],
                   [0,1,0],
                   [0,LaurentPolynomial([1],-1),LaurentPolynomial([-1],-1)]])
"""