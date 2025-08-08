import json
import random

from burau_representation.Classes.Generators import Generators
from burau_representation.Classes.LaurentMatrix import LaurentMatrix

# ========== CONFIGURATION ==========
NUM_RANDOM_CASES = 10
WORD_LENGTH_RANGE = (50, 500)

# Manual identity words per mod
MANUAL_IDENTITIES = {
    2: [
        "AbAAAAbAAAbAAAbAAAAbAABAABABABAABA",
        "aBaaaaBaaaBaaaBaaaaBaabaabababaaba",
        "BaBBBBaBBBaBBBaBBBBaBBABBABABABBAB",
        "BaabbabababbabbAbbbbAbbbAbbbAbbbbAbbAb",
        "ababbbabbabbabbbabbAbbbAbbAbbAbbbAbA"
    ],
    3: [
        "abaabaababaababaabaababaabaababaabaababaababaabaababaabaababaababaabaababaabaababaabaababaababaabaababaabaababaabaababaababaabaababaabaababaabaababaababaabaababaabaababaababaabaababaabaababaabaababaababaabaabaBaBaBBaBBaBaBBaBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBaBBaBaBBaBBaBaBBaBaBBaBaBBaBBaBaBBaBaBBaBaBBaBBaBaBBaBaBBaBBaBaBBaBaBBaBaBBaBBaBaB"
    ],
    5: [],
    None: []
}

MODS = [2, 3, 5, None]


def generate_random_word(length):
    letters = ['A', 'B', 'a', 'b']

    return ''.join(random.choices(letters, k=length))


def compute_product(word, gens, mod):
    result = LaurentMatrix.identity(mod)

    for g in word:
        result = result * gens[g]
    return result


def is_identity(matrix):
    identity = LaurentMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], modulo=matrix.modulo)
    for i in range(3):
        for j in range(3):
            if matrix.matrix[i, j].coefficients.tolist() != identity.matrix[i, j].coefficients.tolist() or matrix.matrix[i, j].min_power != identity.matrix[i, j].min_power:
                return False
    return True


def save_case(word, result, mod):
    mod_label = f"mod{mod}" if mod is not None else "nomod"
    filename = f"random_tests_{mod_label}.jsonl"
    entry = {
        "word": word,
        "result": result.to_nested_list()
    }
    with open(filename, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"Saved to {filename}: len={len(word)}")


# ========== MAIN PROCESS ==========
def main():
    for mod in MODS:
        gens = Generators(mod)

        # Add manual identity words
        manual_words = MANUAL_IDENTITIES.get(mod, [])
        for word in manual_words:
            result = compute_product(word, gens, mod)
            if is_identity(result):
                save_case(word, result, mod)
            else:
                print(f"[WARNING] Manual word is not identity (mod {mod}): {word}")

        # Add random test cases
        for _ in range(NUM_RANDOM_CASES):
            length = random.randint(*WORD_LENGTH_RANGE)
            word = generate_random_word(length)
            result = compute_product(word, gens, mod)
            save_case(word, result, mod)


if __name__ == '__main__':
    main()