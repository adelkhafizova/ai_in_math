from burau_package.classes.laurent_matrix import LaurentMatrix
from burau_package.classes.laurent_polynomial import LaurentPolynomial


class GeneratorElement:
    def __init__(self, name: str, mod):
        self.name = name
        self.mod = mod

        # ─────────── Generator A ───────────
        if name == 'A':
            # 3×3 Burau matrix A
            #   [  0,        0,       (−x⁻¹)    ]
            #   [  0,      (−x),   (−x⁻¹ + x)    ]
            #   [ (−1),       0,   (−x⁻¹ + 1)    ]
            self.matrix = LaurentMatrix([
                [0, 0, LaurentPolynomial([-1], -1, mod)],
                [0, LaurentPolynomial([-1], 1, mod), LaurentPolynomial([-1, 0, 1], -1, mod)],
                [LaurentPolynomial([-1], 0, mod), 0, LaurentPolynomial([-1, 1], -1, mod)]
            ], mod)

            # Precompute every constant LaurentPolynomial used by A's multiply:
            self._A_c_r00 = LaurentPolynomial([-1], 0, mod)  # (−1)
            self._A_c_r01 = LaurentPolynomial([-1], 1, mod)  # (−x)
            self._A_c_r02a = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._A_c_r02b = LaurentPolynomial([-1, 0, 1], -1, mod)  # (−x⁻¹ + x)
            self._A_c_r02c = LaurentPolynomial([-1, 1], -1, mod)  # (−x⁻¹ + 1)

            self._A_c_r10 = LaurentPolynomial([-1], 0, mod)  # (−1)
            self._A_c_r11 = LaurentPolynomial([-1], 1, mod)  # (−x)
            self._A_c_r12a = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._A_c_r12b = LaurentPolynomial([-1, 0, 1], -1, mod)  # (−x⁻¹ + x)
            self._A_c_r12c = LaurentPolynomial([-1, 1], -1, mod)  # (−x⁻¹ + 1)

            self._A_c_r20 = LaurentPolynomial([-1], 0, mod)  # (−1)
            self._A_c_r21 = LaurentPolynomial([-1], 1, mod)  # (−x)
            self._A_c_r22a = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._A_c_r22b = LaurentPolynomial([-1, 0, 1], -1, mod)  # (−x⁻¹ + x)
            self._A_c_r22c = LaurentPolynomial([-1, 1], -1, mod)  # (−x⁻¹ + 1)

            self.mul = self._right_multiply_A

        # ─────────── Generator a (inverse of A) ───────────
        elif name == 'a':
            # 3×3 Burau matrix a = A⁻¹
            #   [ (1 − x),    0,    (−1)   ]
            #   [ (x⁻¹ − x), (−x⁻¹),  0    ]
            #   [ (−x),       0,      0   ]
            self.matrix = LaurentMatrix([
                [LaurentPolynomial([1, -1], 0, mod), 0, LaurentPolynomial([-1], 0, mod)],
                [LaurentPolynomial([1, 0, -1], -1, mod), LaurentPolynomial([-1], -1, mod), 0],
                [LaurentPolynomial([-1], 1, mod), 0, 0]
            ], mod)

            # Precompute every constant LaurentPolynomial used by a's multiply:
            self._a_c_00 = LaurentPolynomial([1, -1], 0, mod)  # (1 − x)
            self._a_c_02 = LaurentPolynomial([-1], 0, mod)  # (−1)

            self._a_c_10 = LaurentPolynomial([1, 0, -1], -1, mod)  # (x⁻¹ − x)
            self._a_c_11 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)

            self._a_c_20 = LaurentPolynomial([-1], 1, mod)  # (−x)

            self.mul = self._right_multiply_a

        # ─────────── Generator B ───────────
        elif name == 'B':
            # 3×3 Burau matrix B
            #   [ (−x⁻¹),  1,  0 ]
            #   [   0,     1,  0 ]
            #   [   0,     1, (−x) ]
            self.matrix = LaurentMatrix([
                [LaurentPolynomial([-1], -1, mod), 1, 0],
                [0, 1, 0],
                [0, 1, LaurentPolynomial([-1], 1, mod)]
            ], mod)

            # Precompute every constant LaurentPolynomial used by B's multiply:
            self._B_c_r00 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._B_c_r02 = LaurentPolynomial([-1], 1, mod)  # (−x)

            self._B_c_r10 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._B_c_r12 = LaurentPolynomial([-1], 1, mod)  # (−x)

            self._B_c_r20 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)
            self._B_c_r22 = LaurentPolynomial([-1], 1, mod)  # (−x)

            self.mul = self._right_multiply_B

        # ─────────── Generator b (inverse of B) ───────────
        elif name == 'b':
            # 3×3 Burau matrix b = B⁻¹
            #   [ (−x),      x,      0    ]
            #   [   0,       1,      0    ]
            #   [   0,      x⁻¹,   (−x⁻¹) ]
            self.matrix = LaurentMatrix([
                [LaurentPolynomial([-1], 1, mod), LaurentPolynomial([1], 1, mod), 0],
                [0, 1, 0],
                [0, LaurentPolynomial([1], -1, mod), LaurentPolynomial([-1], -1, mod)]
            ], mod)

            # Precompute every constant LaurentPolynomial used by b's multiply:
            self._b_c_00 = LaurentPolynomial([-1], 1, mod)  # (−x)
            self._b_c_01a = LaurentPolynomial([1], 1, mod)  # x
            self._b_c_01b = LaurentPolynomial([1], -1, mod)  # x⁻¹
            self._b_c_02 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)

            self._b_c_21 = LaurentPolynomial([1], -1, mod)  # ( x⁻¹)
            self._b_c_22 = LaurentPolynomial([-1], -1, mod)  # (−x⁻¹)

            self.mul = self._right_multiply_b

        else:
            raise ValueError(f"Unknown generator name: {name}")

    def __rmul__(self, other):
        if not isinstance(other, LaurentMatrix):
            raise TypeError(f"Cannot multiply {type(other)} by GeneratorElement '{self.name}'")
        return self.mul(other)

    # ─────────────────────────────────────────────────────────────────────────
    # Right‐multiply by A:  R = M · A
    # ─────────────────────────────────────────────────────────────────────────
    def _right_multiply_A(self, M: LaurentMatrix):
        mod = self.mod
        m = M.matrix

        # Row 0:
        r00 = self._A_c_r00 * m[0, 2]  # (−1)*m[0,2]
        r01 = self._A_c_r01 * m[0, 1]  # (−x)*m[0,1]
        r02 = (
                self._A_c_r02a * m[0, 0]  # (−x⁻¹) * m[0,0]
                + self._A_c_r02b * m[0, 1]  # (−x⁻¹ + x) * m[0,1]
                + self._A_c_r02c * m[0, 2]  # (−x⁻¹ + 1) * m[0,2]
        )

        # Row 1:
        r10 = self._A_c_r10 * m[1, 2]  # (−1)*m[1,2]
        r11 = self._A_c_r11 * m[1, 1]  # (−x)*m[1,1]
        r12 = (
                self._A_c_r12a * m[1, 0]  # (−x⁻¹)*m[1,0]
                + self._A_c_r12b * m[1, 1]  # (−x⁻¹ + x)*m[1,1]
                + self._A_c_r12c * m[1, 2]  # (−x⁻¹ + 1)*m[1,2]
        )

        # Row 2:
        r20 = self._A_c_r20 * m[2, 2]  # (−1)*m[2,2]
        r21 = self._A_c_r21 * m[2, 1]  # (−x)*m[2,1]
        r22 = (
                self._A_c_r22a * m[2, 0]  # (−x⁻¹)*m[2,0]
                + self._A_c_r22b * m[2, 1]  # (−x⁻¹ + x)*m[2,1]
                + self._A_c_r22c * m[2, 2]  # (−x⁻¹ + 1)*m[2,2]
        )

        return LaurentMatrix([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ], mod)

    # ─────────────────────────────────────────────────────────────────────────
    # Right‐multiply by a:  R = M · a
    # ─────────────────────────────────────────────────────────────────────────
    def _right_multiply_a(self, M: LaurentMatrix):
        mod = self.mod
        m = M.matrix

        #   a = [
        #     [ (1 − x),    0,     (−1)    ],
        #     [ (x⁻¹ − x), (−x⁻¹),   0    ],
        #     [ (−x),        0,      0   ]
        #   ]

        # Row 0:
        r00 = (
                self._a_c_00 * m[0, 0]  # (1 − x)*m[0,0]
                + self._a_c_10 * m[0, 1]  # (x⁻¹ − x)*m[0,1]
                + self._a_c_20 * m[0, 2]  # (−x)*m[0,2]
        )
        r01 = self._a_c_11 * m[0, 1]  # (−x⁻¹)*m[0,1]
        r02 = self._a_c_02 * m[0, 0]  # (−1)*m[0,0]

        # Row 1:
        r10 = (
                self._a_c_00 * m[1, 0]  # (1 − x)*m[1,0]
                + self._a_c_10 * m[1, 1]  # (x⁻¹ − x)*m[1,1]
                + self._a_c_20 * m[1, 2]  # (−x)*m[1,2]
        )
        r11 = self._a_c_11 * m[1, 1]  # (−x⁻¹)*m[1,1]
        r12 = self._a_c_02 * m[1, 0]  # (−1)*m[1,0]

        # Row 2:
        r20 = (
                self._a_c_00 * m[2, 0]  # (1 − x)*m[2,0]
                + self._a_c_10 * m[2, 1]  # (x⁻¹ − x)*m[2,1]
                + self._a_c_20 * m[2, 2]  # (−x)*m[2,2]
        )
        r21 = self._a_c_11 * m[2, 1]  # (−x⁻¹)*m[2,1]
        r22 = self._a_c_02 * m[2, 0]  # (−1)*m[2,0]

        return LaurentMatrix([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ], mod)

    # ─────────────────────────────────────────────────────────────────────────
    # Right‐multiply by B:  R = M · B
    # ─────────────────────────────────────────────────────────────────────────
    def _right_multiply_B(self, M: LaurentMatrix):
        mod = self.mod
        m = M.matrix

        #   B = [
        #     [ (−x⁻¹),   1,    0    ],
        #     [   0,      1,    0    ],
        #     [   0,      1,   (−x)  ]
        #   ]

        # Row 0:
        r00 = self._B_c_r00 * m[0, 0]  # (−x⁻¹)*m[0,0]
        r01 = m[0, 0] + m[0, 1] + m[0, 2]  # m[0,0] + m[0,1] + m[0,2]
        r02 = self._B_c_r02 * m[0, 2]  # (−x)*m[0,2]

        # Row 1:
        r10 = self._B_c_r10 * m[1, 0]  # (−x⁻¹)*m[1,0]
        r11 = m[1, 0] + m[1, 1] + m[1, 2]  # m[1,0] + m[1,1] + m[1,2]
        r12 = self._B_c_r12 * m[1, 2]  # (−x)*m[1,2]

        # Row 2:
        r20 = self._B_c_r20 * m[2, 0]  # (−x⁻¹)*m[2,0]
        r21 = m[2, 0] + m[2, 1] + m[2, 2]  # m[2,0] + m[2,1] + m[2,2]
        r22 = self._B_c_r22 * m[2, 2]  # (−x)*m[2,2]

        return LaurentMatrix([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ], mod)

    # ─────────────────────────────────────────────────────────────────────────
    # Right‐multiply by b (inverse of B):  R = M · b
    # ─────────────────────────────────────────────────────────────────────────
    def _right_multiply_b(self, M: LaurentMatrix):
        mod = self.mod
        m = M.matrix

        # b = [
        #   [  (−x),    x,    0    ],
        #   [   0,      1,    0    ],
        #   [   0,    x⁻¹,  (−x⁻¹) ]
        # ]

        # Row 0 of (M·b):
        r00 = self._b_c_00 * m[0, 0]
        # (−x) * m[0,0]
        r01 = (
                self._b_c_01a * m[0, 0]  # x * m[0,0]
                + m[0, 1]  # + 1 * m[0,1]
                + self._b_c_01b * m[0, 2]  # + x⁻¹ * m[0,2]
        )
        r02 = self._b_c_02 * m[0, 2]
        # (−x⁻¹) * m[0,2]

        # ────────────────────────────────────────────────
        #   R[1,0] = m[1,0]*(−x) + m[1,1]*0 + m[1,2]*0
        r10 = self._b_c_00 * m[1, 0]  # (−x)*m[1,0]

        #   R[1,1] = m[1,0]* x  + m[1,1]*1 + m[1,2]* x⁻¹
        r11 = (
                self._b_c_01a * m[1, 0]  # x * m[1,0]
                + m[1, 1]  # + 1 * m[1,1]
                + self._b_c_01b * m[1, 2]  # + x⁻¹ * m[1,2]
        )

        #   R[1,2] = m[1,0]*0 + m[1,1]*0 + m[1,2]*(−x⁻¹)
        r12 = self._b_c_02 * m[1, 2]  # (−x⁻¹)*m[1,2]

        # ────────────────────────────────────────────────
        # Row 2 of (M·b):
        #   R[2,0] = m[2,0]*(−x) + m[2,1]*0 + m[2,2]*0
        r20 = self._b_c_00 * m[2, 0]  # (−x)*m[2,0]

        #   R[2,1] = m[2,0]* x  + m[2,1]*1 + m[2,2]* x⁻¹
        r21 = (
                self._b_c_01a * m[2, 0]  # x * m[2,0]
                + m[2, 1]  # + 1 * m[2,1]
                + self._b_c_01b * m[2, 2]  # + x⁻¹ * m[2,2]
        )

        #   R[2,2] = m[2,0]*0 + m[2,1]*0 + m[2,2]*(−x⁻¹)
        r22 = self._b_c_02 * m[2, 2]  # (−x⁻¹)*m[2,2]

        return LaurentMatrix([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ], mod)
