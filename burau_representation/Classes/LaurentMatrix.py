import numpy as np
from burau_representation.Classes.LaurentPolynomial import LaurentPolynomial


class LaurentMatrix:
    def __init__(self, matrix, modulo=None):
        """
        Optimized for 3x3 matrices with large Laurent polynomials.
        matrix: A 2D array where each entry is either a LaurentPolynomial or a number (int/float).
        modulo: If provided, all operations will be performed modulo this value.
        """
        self.modulo = modulo

        # Convert list to numpy array for faster operations
        self.matrix = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                self.matrix[i, j] = self._convert_to_laurent(matrix[i][j])

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

    '''def __mul__(self, other):
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
            raise ValueError("Matrix dimensions must align for multiplication.")'''

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

    def __eq__(self, other):
        """Compare two LaurentMatrix objects entry‐wise."""
        if not isinstance(other, LaurentMatrix):
            return NotImplemented
        if self.modulo != other.modulo:
            return False

        for i in range(3):
            for j in range(3):
                a = self.matrix[i, j]
                b = other.matrix[i, j]
                if a.min_power != b.min_power or a.coefficients.tolist() != b.coefficients.tolist():
                    return False
        return True

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

    @staticmethod
    def identity(mod):
        return LaurentMatrix(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], mod)