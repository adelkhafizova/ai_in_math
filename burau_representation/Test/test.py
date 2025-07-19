import json
import unittest

from burau_representation.Classes.Generators import Generators
from burau_representation.Classes.LaurentMatrix import LaurentMatrix as LaurentMatrix


class TestRandomIdentityCases(unittest.TestCase):
    def run_case_file(self, mod_label, mod_val):
        gens = Generators(mod_val)
        filename = f"random_tests_{mod_label}.jsonl"
        with open(filename) as f:
            for i, line in enumerate(f):
                case = json.loads(line)
                word = case["word"]
                expected = LaurentMatrix.from_nested_list(case["result"])
                result = LaurentMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], modulo=mod_val)

                for g in word:
                    result *= gens[g]

                self.assertTrue(
                    result == expected,
                    # is_same_matrix(result, expected),
                    f"{mod_label} case {i} failed.\nWord: {word}"
                )

    def test_mod2(self):
        self.run_case_file("mod2", 2)

    def test_mod3(self):
        self.run_case_file("mod3", 3)

    def test_mod5(self):
        self.run_case_file("mod5", 5)

    def test_nomod(self):
        self.run_case_file("nomod", None)