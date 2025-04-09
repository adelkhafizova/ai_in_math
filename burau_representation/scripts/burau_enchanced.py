import numpy as np
from itertools import product 

class LaurentPolynomial:
    def __init__(self, coefficients, min_power, modulo=None):
        """
        coefficients: List or array of coefficients.
        min_power: The smallest power of x.
        modulo: If provided, all operations will be performed modulo this value.
        """
        self.coefficients = np.array(coefficients, dtype=int if modulo else float)
        self.min_power = int(min_power)
        self.modulo = modulo
        
        # If modulo is provided, apply it to coefficients immediately
        if self.modulo is not None:
            self.coefficients = np.mod(self.coefficients, self.modulo)
        
        # Trim immediately to optimize for large polynomials
        self.trim()

    def trim(self):
        """
        Remove leading and trailing zeros from the coefficients.
        Adjust min_power accordingly.
        """
        if len(self.coefficients) == 0:
            self.coefficients = np.array([0])
            self.min_power = 0
            return
            
        # Find the first and last non-zero coefficients - optimize for large arrays
        # Use np.flatnonzero instead of np.nonzero for slightly better performance
        non_zero_indices = np.flatnonzero(self.coefficients)
        
        if len(non_zero_indices) == 0:
            # Polynomial is zero
            self.coefficients = np.array([0])
            self.min_power = 0
        else:
            start, end = non_zero_indices[0], non_zero_indices[-1]
            # Only trim if necessary to avoid unnecessary array copying
            if start > 0 or end < len(self.coefficients) - 1:
                self.coefficients = self.coefficients[start:end + 1]
                self.min_power += int(start)

    def _convert_to_laurent(self, entry):
        if isinstance(entry, LaurentPolynomial):
            # Make sure modulo properties match
            if entry.modulo != self.modulo:
                result = LaurentPolynomial(entry.coefficients.copy(), entry.min_power, self.modulo)
                return result
            return entry
        elif isinstance(entry, (int, float)):
            # Optimize common cases
            if entry == 0:
                return LaurentPolynomial([0], 0, self.modulo)
            elif entry == 1:
                return LaurentPolynomial([1], 0, self.modulo)
            else:
                return LaurentPolynomial([entry], 0, self.modulo)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")

    def __call__(self, x):
        """
        Evaluate the Laurent polynomial at a given x.
        Using Horner's method for more efficient evaluation of large polynomials.
        """
        if len(self.coefficients) == 1:
            return self.coefficients[0]
            
        # For very large polynomials, using Horner's method
        x_val = float(x)
        if len(self.coefficients) > 100:  # Threshold for using Horner's method
            # First compute x^min_power
            x_min = x_val ** self.min_power
            
            # Then use Horner's method for the rest
            result = self.coefficients[-1]
            for c in self.coefficients[-2::-1]:
                result = result * x_val + c
            result = result * x_min
        else:
            # For smaller polynomials, the vectorized approach is faster
            powers = np.arange(self.min_power, self.min_power + len(self.coefficients))
            result = np.sum(self.coefficients * (x_val ** powers))
            
        if self.modulo is not None:
            result = result % self.modulo
            
        return result

    def __mul__(self, other):
        """
        Optimized multiplication of Laurent polynomials focused on large polynomials.
        """
        if isinstance(other, (int, float)):
            # Fast path for scalar multiplication
            if other == 0:
                return LaurentPolynomial([0], 0, self.modulo)
            elif other == 1:
                # Return a copy to prevent unintended modifications
                return LaurentPolynomial(self.coefficients.copy(), self.min_power, self.modulo)
                
            other_val = other if self.modulo is None else other % self.modulo
            new_coeffs = self.coefficients * other_val
            
            if self.modulo is not None:
                new_coeffs = np.mod(new_coeffs, self.modulo)
                
            return LaurentPolynomial(new_coeffs, self.min_power, self.modulo)
        
        # Convert to Laurent polynomial if needed
        other_converted = self._convert_to_laurent(other)
        
        # Check for special cases (optimization for large polynomials)
        if len(other_converted.coefficients) == 1:
            scalar = other_converted.coefficients[0]
            if scalar == 0:
                return LaurentPolynomial([0], 0, self.modulo)
            elif scalar == 1:
                # Handle power shift without creating new arrays
                return LaurentPolynomial(
                    self.coefficients.copy(), 
                    self.min_power + other_converted.min_power, 
                    self.modulo
                )
        

        new_min_power = int(self.min_power + other_converted.min_power)
        new_coefficients = np.convolve(self.coefficients, other_converted.coefficients)
        
        if self.modulo is not None:
            new_coefficients = np.mod(new_coefficients, self.modulo)
            
        result = LaurentPolynomial(new_coefficients, new_min_power, self.modulo)
        result.trim()  # Important for large polynomials to keep size manageable
        return result
        
    def to_array(self):
        """Convert to a compact array representation: [coefficients, min_power, modulo]."""
        return [self.coefficients.tolist(), int(self.min_power), self.modulo]
    
    @staticmethod
    def from_array(array):
        """Convert from a compact array representation: [coefficients, min_power, modulo]."""
        modulo = None if len(array) <= 2 or array[2] is None else array[2]
        return LaurentPolynomial(array[0], array[1], modulo)

    def __add__(self, other):
        """Optimized addition for large polynomials."""
        if isinstance(other, (int, float)):
            # Optimize common scalar cases
            if other == 0:
                return LaurentPolynomial(self.coefficients.copy(), self.min_power, self.modulo)
            
            # Fast path for adding a scalar
            other_converted = LaurentPolynomial([other], 0, self.modulo)
        else:
            # Ensure other has the same modulo
            other_converted = self._convert_to_laurent(other)
        
        # Check if we're adding zero polynomial
        if len(other_converted.coefficients) == 1 and other_converted.coefficients[0] == 0:
            return LaurentPolynomial(self.coefficients.copy(), self.min_power, self.modulo)
        
        # Efficiently handle large polynomials with different powers
        min_power = min(self.min_power, other_converted.min_power)
        max_self_power = self.min_power + len(self.coefficients) - 1
        max_other_power = other_converted.min_power + len(other_converted.coefficients) - 1
        max_power = max(max_self_power, max_other_power)
        
        # Create an array for the new coefficients (1 more than degree difference)
        new_length = max_power - min_power + 1
        new_coefficients = np.zeros(new_length, dtype=int if self.modulo else float)
        
        # Add self's coefficients at correct positions
        self_start = self.min_power - min_power
        self_end = self_start + len(self.coefficients)
        new_coefficients[self_start:self_end] += self.coefficients
        
        # Add other's coefficients at correct positions
        other_start = other_converted.min_power - min_power
        other_end = other_start + len(other_converted.coefficients)
        new_coefficients[other_start:other_end] += other_converted.coefficients
        
        if self.modulo is not None:
            new_coefficients = np.mod(new_coefficients, self.modulo)
            
        result = LaurentPolynomial(new_coefficients, min_power, self.modulo)
        result.trim()  # Critical for large polynomials
        return result

    def __str__(self):
        """
        String representation of the Laurent polynomial.
        """
        terms = []
        powers = np.arange(self.min_power, self.min_power + len(self.coefficients))
        for coef, power in zip(self.coefficients, powers):
            if coef != 0:
                term = f"{coef}x^{power}" if power != 0 else str(coef)
                terms.append(term)
        
        result = " + ".join(terms[::-1]) or "0"
        if self.modulo is not None:
            result += f" (mod {self.modulo})"
        return result

    def is_zero(self):
        """Check if this polynomial is zero (optimized for large polynomials)"""
        return len(self.coefficients) == 1 and self.coefficients[0] == 0

    def is_one(self):
        """Check if this polynomial is 1 (optimized for large polynomials)"""
        return (len(self.coefficients) == 1 and 
                self.coefficients[0] == 1 and 
                self.min_power == 0)


class LaurentMatrix:
    def __init__(self, matrix, modulo=None):
        """
        Optimized for 3x3 matrices with large Laurent polynomials.
        matrix: A 2D array where each entry is either a LaurentPolynomial or a number (int/float).
        modulo: If provided, all operations will be performed modulo this value.
        """
        self.modulo = modulo
        
        # Optimized for 3x3 matrices with large polynomials
        # We assume fixed 3x3 size so no need to dynamically determine dimensions
        if isinstance(matrix, list):
            # Convert list to numpy array for faster operations
            self.matrix = np.empty((3, 3), dtype=object)
            
            for i in range(3):
                for j in range(3):
                    self.matrix[i, j] = self._convert_to_laurent(matrix[i][j])
        else:
            # Assuming matrix is already a numpy array
            self.matrix = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    self.matrix[i, j] = self._convert_to_laurent(matrix[i, j])

    def _convert_to_laurent(self, entry):
        if isinstance(entry, LaurentPolynomial):
            # Ensure the polynomial has the same modulo as the matrix
            if entry.modulo != self.modulo:
                return LaurentPolynomial(entry.coefficients.copy(), entry.min_power, self.modulo)
            return entry
        elif isinstance(entry, (int, float)):
            # Optimized conversion for common cases
            if entry == 0:
                return LaurentPolynomial([0], 0, self.modulo)
            elif entry == 1:
                return LaurentPolynomial([1], 0, self.modulo)
            elif entry == -1:
                return LaurentPolynomial([-1], 0, self.modulo)
            else:
                return LaurentPolynomial([entry], 0, self.modulo)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")

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
                    sum_poly = LaurentPolynomial([0], 0, self.modulo)
                    
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
                        result[i, j] = LaurentPolynomial([1], 0, self.modulo)
                    else:
                        result[i, j] = LaurentPolynomial([0], 0, self.modulo)
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
                    result[i, j] = LaurentPolynomial([1], 0, self.modulo)
                else:
                    result[i, j] = LaurentPolynomial([0], 0, self.modulo)
        
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
    
    def to_nested_list(self):
        """Convert LaurentMatrix to a serializable format (nested list)."""
        nested_list = [[entry.to_array() for entry in row] for row in self.matrix]
        return {"matrix": nested_list, "modulo": self.modulo}
    
    def convert_to_modulo(self, p):
        """
        Convert the current matrix to a new matrix with operations performed modulo p.
        """
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

    @staticmethod
    def from_nested_list(data):
        """Convert from a nested list of compact polynomial representations."""
        if isinstance(data, dict):
            nested_list = data["matrix"]
            modulo = data.get("modulo", None)
        else:
            nested_list = data
            modulo = None
            
        matrix = [[LaurentPolynomial.from_array(entry) for entry in row] for row in nested_list]
        return LaurentMatrix(matrix, modulo)


# Function to benchmark multiplication performance
def benchmark_multiplication(A, B, repetitions=100):
    """Benchmark the multiplication of two Laurent matrices"""
    import time
    
    start_time = time.time()
    for _ in range(repetitions):
        result = A * B
    end_time = time.time()
    
    return (end_time - start_time) / repetitions


A = LaurentMatrix([[0,0,LaurentPolynomial([-1],-1)],
                   [0,LaurentPolynomial([-1],1),LaurentPolynomial([-1,0,1],-1)],
                   [-1,0,LaurentPolynomial([-1,1],-1)]])

B = LaurentMatrix([[LaurentPolynomial([-1],-1),1,0],
                   [0,1,0],
                   [0,1,LaurentPolynomial([-1],1)]])

a = LaurentMatrix([[LaurentPolynomial([1,-1],0),0,-1],
                   [LaurentPolynomial([1,0,-1],-1),LaurentPolynomial([-1],-1),0],
                   [LaurentPolynomial([-1],1),0,0]])

b = LaurentMatrix([[LaurentPolynomial([-1],1),LaurentPolynomial([1],1),0],
                   [0,1,0],
                   [0,LaurentPolynomial([1],-1),LaurentPolynomial([-1],-1)]])
