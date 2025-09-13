from itertools import product
import numpy as np

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
            # Make sure modulo properties match
            if entry.modulo != self.modulo:
                result = LaurentPolynomial(entry.coefficients.copy(), entry.min_power, self.modulo)
                return result
            return entry
        elif isinstance(entry, (int, float)):
            return LaurentPolynomial([entry], 0, self.modulo)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")

    def __call__(self, x):
        """
        Evaluate the Laurent polynomial at a given x.
        """
        x = float(x)
        powers = np.arange(self.min_power, self.min_power + len(self.coefficients))
        result = np.sum(self.coefficients * (x ** powers))
        if self.modulo is not None:
            result = result % self.modulo
        return result

    def __mul__(self, other):
        """
        Multiplication of two Laurent polynomials. 
        If one multiplicant is number then multiplying polynomial by number.
        """
        if isinstance(other, LaurentPolynomial):
            # Ensure other has the same modulo
            other_converted = self._convert_to_laurent(other)
            
            new_min_power = int(self.min_power + other_converted.min_power)
            new_coefficients = np.convolve(self.coefficients, other_converted.coefficients)
            
            if self.modulo is not None:
                new_coefficients = np.mod(new_coefficients, self.modulo)
                
            result = LaurentPolynomial(new_coefficients, new_min_power, self.modulo)
            result.trim()
            return result
        elif isinstance(other, (int, float)):
            other_val = other
            if self.modulo is not None:
                other_val = other % self.modulo
            
            result = LaurentPolynomial(self.coefficients * other_val, int(self.min_power), self.modulo)
            
            if self.modulo is not None:
                result.coefficients = np.mod(result.coefficients, self.modulo)
                
            result.trim()
            return result
        else:
            raise ValueError("Unsupported type for multiplication.")
        
    def to_array(self):
        """Convert to a compact array representation: [coefficients, min_power, modulo]."""
        return [self.coefficients.tolist(), int(self.min_power), self.modulo]
    
    @staticmethod
    def from_array(array):
        """Convert from a compact array representation: [coefficients, min_power, modulo]."""
        modulo = None if len(array) <= 2 or array[2] is None else array[2]
        return LaurentPolynomial(array[0], array[1], modulo)

    def __add__(self, other):
        if isinstance(other, LaurentPolynomial):
            # Ensure other has the same modulo
            other_converted = self._convert_to_laurent(other)
            
            # Align the powers of the two polynomials
            min_power = min(self.min_power, other_converted.min_power)
            max_power = max(self.min_power + len(self.coefficients), 
                           other_converted.min_power + len(other_converted.coefficients))
            
            # Create an array for the new coefficients
            new_coefficients = np.zeros(max_power - min_power, dtype=int if self.modulo else float)
            
            # Add self's coefficients
            self_offset = self.min_power - min_power
            new_coefficients[self_offset:self_offset + len(self.coefficients)] += self.coefficients
            
            # Add other's coefficients
            other_offset = other_converted.min_power - min_power
            new_coefficients[other_offset:other_offset + len(other_converted.coefficients)] += other_converted.coefficients
            
            if self.modulo is not None:
                new_coefficients = np.mod(new_coefficients, self.modulo)
                
            result = LaurentPolynomial(new_coefficients, min_power, self.modulo)
            result.trim()
            return result
        
        elif isinstance(other, (int, float)):
            return self + self._convert_to_laurent(other)
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
        
        result = " + ".join(terms[::-1]) or "0"
        if self.modulo is not None:
            result += f" (mod {self.modulo})"
        return result


class LaurentMatrix:
    def __init__(self, matrix, modulo=None):
        """
        matrix: A 2D array where each entry is either a LaurentPolynomial or a number (int/float).
        modulo: If provided, all operations will be performed modulo this value.
        """
        self.modulo = modulo
        self.matrix = np.array([[self._convert_to_laurent(entry) for entry in row] for row in matrix])

    def _convert_to_laurent(self, entry):
        if isinstance(entry, LaurentPolynomial):
            # Ensure the polynomial has the same modulo as the matrix
            if entry.modulo != self.modulo:
                return LaurentPolynomial(entry.coefficients.copy(), entry.min_power, self.modulo)
            return entry
        elif isinstance(entry, (int, float)):
            return LaurentPolynomial([entry], 0, self.modulo)
        else:
            raise ValueError("Matrix entries must be either Laurent polynomials or numbers.")

    def __add__(self, other):
        if isinstance(other, LaurentMatrix) and self.matrix.shape == other.matrix.shape:
            # Create a new matrix with the same modulo as self
            other_with_modulo = LaurentMatrix(other.matrix, self.modulo)
            
            result = [
                [self.matrix[i, j] + other_with_modulo.matrix[i, j] for j in range(self.matrix.shape[1])]
                for i in range(self.matrix.shape[0])
            ]
            return LaurentMatrix(result, self.modulo)
        else:
            raise ValueError("Matrix dimensions must match for addition.")

    def __mul__(self, other):
        if isinstance(other, LaurentMatrix) and self.matrix.shape[1] == other.matrix.shape[0]:
            # Ensure other has the same modulo
            other_with_modulo = LaurentMatrix(other.matrix, self.modulo)
            
            rows, cols = self.matrix.shape[0], other_with_modulo.matrix.shape[1]
            common_dim = self.matrix.shape[1]

            # Initialize the result matrix with zero Laurent polynomials
            result = [[LaurentPolynomial([0], 0, self.modulo) for _ in range(cols)] for _ in range(rows)]

            # Perform matrix multiplication
            for i in range(rows):
                for j in range(cols):
                    for k in range(common_dim):
                        result[i][j] += self.matrix[i, k] * other_with_modulo.matrix[k, j]

            return LaurentMatrix(result, self.modulo)
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

        # Identity matrix with same modulo
        result = LaurentMatrix(np.eye(self.matrix.shape[0], dtype=object), self.modulo)
        
        if n == 0:
            return result
        elif n > 0:
            temp = self
            for _ in range(n):
                result = result * temp
            return result
        else:  # n < 0
            raise ValueError("Negative powers are not supported yet.")  # Implement inverse if needed.

    def __str__(self):
        rows = ["[" + ", ".join(str(entry) for entry in row) + "]" for row in self.matrix]
        result = "[\n " + ",\n ".join(rows) + "\n]"
        if self.modulo is not None:
            result = f"Matrix (mod {self.modulo}):\n" + result
        return result
    
    def __call__(self, x):
        """Evaluate the matrix at a specific value of x."""
        result = np.matrix([[polynomial(x) for polynomial in row] for row in self.matrix])
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
        
        Parameters:
            p (int): The modulus to apply
            
        Returns:
            LaurentMatrix: A new matrix with the same entries but operations performed modulo p
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError("Modulus must be a positive integer.")
            
        # Create a new matrix with the modulo set to p
        new_matrix = [
            [
                LaurentPolynomial(
                    entry.coefficients.copy(), 
                    entry.min_power, 
                    p
                ) for entry in row
            ] for row in self.matrix
        ]
        
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
            
        return LaurentMatrix([[LaurentPolynomial.from_array(entry) for entry in row] for row in nested_list], modulo)
    
    


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


