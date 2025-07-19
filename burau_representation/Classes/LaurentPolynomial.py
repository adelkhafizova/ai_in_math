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
        Multiplication of Laurent polynomials focused on large polynomials.
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