
import sympy as sp
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly

class PolynomialMatrix:
    """
    A class for 3x3 matrices with entries being sympy polynomials over a single variable,
    with support for modular arithmetic.
    """
    
    def __init__(self, matrix=None, var=None, modulus=None):
        """
        Initialize a 3x3 matrix with single-variable polynomial entries.
        
        Parameters:
        -----------
        matrix : list or numpy array, optional
            A 3x3 matrix with polynomial entries. If None, initializes with zeros.
        var : sympy.Symbol, optional
            The variable for the polynomials. If None, uses 'x'.
        modulus : int, optional
            The modulus for polynomial arithmetic. If None, standard arithmetic is used.
        """
        # Initialize the variable for the polynomials
        self.var = sp.symbols('x') if var is None else var
        self.modulus = modulus
        
        if matrix is None:
            # Initialize with zeros
            self.matrix = [[Poly(0, self.var) for _ in range(3)] for _ in range(3)]
        else:
            # Initialize matrix
            self.matrix = [[None for _ in range(3)] for _ in range(3)]
            
            # Verify dimensions
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("Matrix must be 3x3")
            
            # Convert entries to polynomials
            for i in range(3):
                for j in range(3):
                    entry = matrix[i][j]
                    
                    # Convert to polynomial form
                    if not isinstance(entry, Poly):
                        if isinstance(entry, sp.Expr):
                            poly = Poly(entry, self.var)
                        else:
                            # Convert integers, etc.
                            poly = Poly(sp.sympify(entry), self.var)
                    else:
                        # Already a polynomial
                        poly = entry
                    
                    # Apply modulus if specified
                    if self.modulus is not None:
                        poly = self._apply_modulus(poly)
                    
                    self.matrix[i][j] = poly
    
    def _apply_modulus(self, poly):
        """Apply the modulus to coefficients of the polynomial."""
        if self.modulus is None:
            return poly
        
        # Get the polynomial coefficients
        coeffs = poly.all_coeffs()
        # Apply modulus to each coefficient
        mod_coeffs = [coeff % self.modulus for coeff in coeffs]
        
        # Create a new polynomial with modular coefficients
        return Poly.from_list(mod_coeffs, self.var)
    
    def __str__(self):
        """String representation of the matrix."""
        rows = []
        for i in range(3):
            row = ["  " + str(self.matrix[i][j].as_expr()) for j in range(3)]
            rows.append("[" + ", ".join(row) + "  ]")
        return "[\n" + "\n".join(rows) + "\n]"
    
    def __repr__(self):
        """Formal string representation of the matrix."""
        return f"PolynomialMatrix({self.matrix}, var={self.var}, modulus={self.modulus})"
    
    def __mul__(self, other):
        """
        Multiply the matrix by another matrix.
        
        Parameters:
        -----------
        other : PolynomialMatrix or scalar
            The matrix or scalar to multiply with.
        
        Returns:
        --------
        PolynomialMatrix: The result of the multiplication.
        """
        # Scalar multiplication
        if isinstance(other, (int, float, sp.Expr)):
            result = [[None for _ in range(3)] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    scalar_poly = Poly(other, self.var)
                    if self.modulus is not None:
                        scalar_poly = self._apply_modulus(scalar_poly)
                    result[i][j] = self.matrix[i][j] * scalar_poly
            
            return PolynomialMatrix(result, self.var, self.modulus)
        
        # Matrix multiplication
        elif isinstance(other, PolynomialMatrix):
            # Check if variables match
            if self.var != other.var:
                raise ValueError("Both matrices must use the same variable")
            
            # Determine the modulus for the result
            result_modulus = self.modulus
            if self.modulus != other.modulus:
                # Only use modulus if both matrices have the same modulus
                if self.modulus is not None and other.modulus is not None:
                    raise ValueError("Both matrices must use the same modulus")
                result_modulus = self.modulus if self.modulus is not None else other.modulus
            
            # Perform matrix multiplication
            result = [[None for _ in range(3)] for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    sum_poly = Poly(0, self.var)
                    for k in range(3):
                        prod = self.matrix[i][k] * other.matrix[k][j]
                        sum_poly += prod
                    
                    # Apply modulus if needed
                    if result_modulus is not None:
                        sum_poly = self._apply_modulus(sum_poly)
                    
                    result[i][j] = sum_poly
            
            return PolynomialMatrix(result, self.var, result_modulus)
        
        else:
            raise TypeError("Multiplication is only defined with PolynomialMatrix or scalar")
    
    def __rmul__(self, other):
        """
        Right multiplication with a scalar.
        
        Parameters:
        -----------
        other : scalar
            The scalar to multiply with.
        
        Returns:
        --------
        PolynomialMatrix: The result of the multiplication.
        """
        if isinstance(other, (int, float, sp.Expr)):
            return self.__mul__(other)
        else:
            raise TypeError("Right multiplication is only defined with scalars")


class MatrixWordCalculator:
    """
    A class to compute products of words in matrices A, B, a, b where all four matrices
    are specified manually.
    """
    
    def __init__(self, A, B, a, b, var=None, modulus=None):
        """
        Initialize with four matrices A, B, a, and b.
        
        Parameters:
        -----------
        A, B, a, b : PolynomialMatrix or list
            The four matrices to use in word calculations.
        var : sympy.Symbol, optional
            The variable for the polynomials. If None, uses 'x'.
        modulus : int, optional
            The modulus for polynomial arithmetic. If None, standard arithmetic is used.
        """
        self.var = sp.symbols('x') if var is None else var
        self.modulus = modulus
        
        # Convert all inputs to PolynomialMatrix if they're not already
        if not isinstance(A, PolynomialMatrix):
            self.A = PolynomialMatrix(A, self.var, self.modulus)
        else:
            self.A = A
            
        if not isinstance(B, PolynomialMatrix):
            self.B = PolynomialMatrix(B, self.var, self.modulus)
        else:
            self.B = B
            
        if not isinstance(a, PolynomialMatrix):
            self.a = PolynomialMatrix(a, self.var, self.modulus)
        else:
            self.a = a
            
        if not isinstance(b, PolynomialMatrix):
            self.b = PolynomialMatrix(b, self.var, self.modulus)
        else:
            self.b = b
    
    def compute_word(self, word):
        """
        Compute the product of a word in A, B, a, b.
        
        Parameters:
        -----------
        word : str
            A string containing only the characters 'A', 'B', 'a', 'b',
            representing a product of matrices.
            For example, "ABaB" means A*B*a*B.
        
        Returns:
        --------
        PolynomialMatrix: The result of the matrix multiplication.
        
        Raises:
        -------
        ValueError: If the word contains invalid characters.
        """
        if not word:
            raise ValueError("Empty word provided")
        
        if not all(c in "ABab" for c in word):
            raise ValueError("Word can only contain 'A', 'B', 'a', 'b'")
        
        # Matrix lookup dictionary
        matrices = {'A': self.A, 'B': self.B, 'a': self.a, 'b': self.b}
        
        # Start with the first matrix
        result = matrices[word[0]]
        
        # Multiply by the rest
        for c in word[1:]:
            result = result * matrices[c]
        
        return result


# Example usage
if __name__ == "__main__":
    # Create symbolic variable
    x = sp.symbols('x')
    
    # Example 1: Create a matrix with polynomial entries
    A = PolynomialMatrix([
        [0, 0, -1],
        [0, -x**2, -1+x**2],
        [-x, 0, -1+x]
    ], var=x, modulus=3)
    B = PolynomialMatrix([
        [-1, x, 0],
        [0, x, 0],
        [0, x, -1*x**2]
    ], var=x, modulus=3)
    a = PolynomialMatrix([
        [x-x**2, 0, -1*x],
        [1-x**2, -1, 0],
        [-1*x**2, 0, 0]
    ], var=x, modulus=3)
    b = PolynomialMatrix([
        [-1*x**2, x**2, 0],
        [0, x, 0],
        [0, 1, -1]
    ], var=x, modulus=3)
    calc = MatrixWordCalculator(A, B, a, b, var=x, modulus=3)
    print(calc.compute_word("aBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaaBaBaaBaaBaBaaBaBaaBaBaaBaaBaBaBABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBABABBABBABABBABABBABBABABBABBABABBABBABABBABABBABBAB"))
    
