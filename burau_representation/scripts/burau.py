from itertools import product
import numpy as np

class LaurentPolynomial:
    def __init__(self, coefficients, min_power):
        """
        coefficients: List or array of coefficients.
        min_power: The smallest power of x.
        """
        self.coefficients = np.array(coefficients)
        self.min_power = int(min_power)

    def trim(self):
        """
        Remove leading and trailing zeros from the coefficients.
        Adjust min_power accordingly.
        """
        if len(self.coefficients) == 0:
            return
        # Find the first and last non-zero coefficients
        non_zero_indices = np.nonzero(self.coefficients)[0]
        if len(non_zero_indices) == 0:
            # Polynomial is zero
            self.coefficients = np.array([0])
            self.min_power = int(0)
        else:
            start, end = non_zero_indices[0], non_zero_indices[-1]
            self.coefficients = self.coefficients[start:end + 1]
            self.min_power += int(start)


    def _convert_to_laurent(self, entry):
        if isinstance(entry, LaurentPolynomial):
            return entry
        elif isinstance(entry, (int, float)):
            return LaurentPolynomial([entry], 0)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")


    def __call__(self, x):
        """
        Evaluate the Laurent polynomial at a given x.
        """
        x = float(x)
        powers = np.arange(self.min_power, self.min_power + len(self.coefficients))
        return np.sum(self.coefficients * (x ** powers))




    def __mul__(self, other):
        """
        Multiplication of two Laurent polynomials. 
        If one multiplicant is number then multiplying polynomial by number.
        """
        if isinstance(other, LaurentPolynomial):
            new_min_power = int(self.min_power + other.min_power)
            new_coefficients = np.convolve(self.coefficients, other.coefficients)
            result = LaurentPolynomial(new_coefficients, new_min_power)
            result.trim()
            return result
        elif isinstance(other, (int, float)):
            result = LaurentPolynomial(self.coefficients * other, int(self.min_power))
            result.trim()
            return result
        else:
            raise ValueError("Unsupported type for multiplication.")
        
    def to_array(self):
        """Convert to a compact array representation: [coefficients, min_power]."""
        return [self.coefficients.tolist(), int(self.min_power)]
    
    @staticmethod
    def from_array(array):
        """Convert from a compact array representation: [coefficients, min_power]."""
        return LaurentPolynomial(array[0], array[1])





    def __add__(self, other):
        if isinstance(other, LaurentPolynomial):
            # Align the powers of the two polynomials
            min_power = min(self.min_power, other.min_power)
            max_power = max(self.min_power + len(self.coefficients), other.min_power + len(other.coefficients))
            
            # Create an array for the new coefficients
            new_coefficients = np.zeros(max_power - min_power)
            
            # Add self's coefficients
            self_offset = self.min_power - min_power
            new_coefficients[self_offset:self_offset + len(self.coefficients)] += self.coefficients
            
            # Add other's coefficients
            other_offset = other.min_power - min_power
            new_coefficients[other_offset:other_offset + len(other.coefficients)] += other.coefficients
            
            result = LaurentPolynomial(new_coefficients, min_power)
            result.trim()
            return result
        
        elif isinstance(other, (int, float)):
            result = self+self._convert_to_laurent(other)
            result.trim()
            return result
        else:
            raise ValueError("Unsupported type for addition.")

        



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
        return " + ".join(terms[::-1]) or "0"
    



class LaurentMatrix:
    def __init__(self, matrix):
        """
        matrix: A 2D array where each entry is either a LaurentPolynomial or a number (int/float).
        """
        self.matrix = np.array([[self._convert_to_laurent(entry) for entry in row] for row in matrix])

    def _convert_to_laurent(self, entry):
        if isinstance(entry, LaurentPolynomial):
            return entry
        elif isinstance(entry, (int, float)):
            return LaurentPolynomial([entry], 0)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")

    def __add__(self, other):
        if isinstance(other, LaurentMatrix) and self.matrix.shape == other.matrix.shape:
            result = [
                [self.matrix[i, j] + other.matrix[i, j] for j in range(self.matrix.shape[1])]
                for i in range(self.matrix.shape[0])
            ]
            return LaurentMatrix(result)
        else:
            raise ValueError("Matrix dimensions must match for addition.")

    def __mul__(self, other):
        if isinstance(other, LaurentMatrix) and self.matrix.shape[1] == other.matrix.shape[0]:
            rows, cols = self.matrix.shape[0], other.matrix.shape[1]
            common_dim = self.matrix.shape[1]

            # Initialize the result matrix with zero Laurent polynomials
            result = [[LaurentPolynomial([0], 0) for _ in range(cols)] for _ in range(rows)]

            # Perform matrix multiplication
            for i in range(rows):
                for j in range(cols):
                    for k in range(common_dim):
                        result[i][j] += self.matrix[i, k] * other.matrix[k, j]

            return LaurentMatrix(result)
        else:
            raise ValueError("Matrix dimensions must align for multiplication.")
        
    def __pow__(self, n):
        """
        Compute the nth power of the LaurentMatrix.
        
        Parameters:
            n (int): The exponent to which the matrix should be raised.
        
        Returns:
            LaurentMatrix: The matrix raised to the nth power.
        """
        if not isinstance(n, int):
            raise ValueError("The exponent must be an integer.")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square to compute powers.")

        # Identity matrix
        result = LaurentMatrix(np.eye(self.matrix.shape[0], dtype=object))
        
        if n == 0:
            return result
        elif n > 0:
            temp = self
            for _ in range(n - 1):
                result = result * temp
            return result
        else:  # n < 0
            raise ValueError("Negative powers are not supported yet.")  # Implement inverse if needed.

    def __str__(self):
        rows = ["[" + ", ".join(str(entry) for entry in row) + "]" for row in self.matrix]
        return "[\n " + ",\n ".join(rows) + "\n]"
    
    def __call__(self,x):
        return np.matrix([[polynomial(x) for polynomial in row] for row in self.matrix])
    
    def to_nested_list(self):
        """Convert LaurentMatrix to a serializable format (nested list)."""
        return [[entry.to_array() for entry in row] for row in self.matrix]
    
    @staticmethod
    def from_nested_list(nested_list):
        """Convert from a nested list of compact polynomial representations."""
        return LaurentMatrix([[LaurentPolynomial.from_array(entry) for entry in row] for row in nested_list])


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