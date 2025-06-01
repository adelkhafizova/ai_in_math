#!/usr/bin/env python3

class FreeGroupWord:
    """Represents a word in a free group."""
    
    def __init__(self, word_str, generators="abcd"):
        """
        Initialize a word in a free group.
        
        Args:
            word_str (str): String representation of the word.
                Lowercase letters represent generators, uppercase represent inverses.
                E.g., "abAc" means a*b*a^(-1)*c
            generators (str): String of available generators
        """
        self.generators = generators
        self.word = self._parse_and_reduce(word_str)
    
    def _parse_and_reduce(self, word_str):
        """Parse the string and perform free reduction."""
        # Map a->1, b->2, etc.; A->-1, B->-2, etc.
        result = []
        for char in word_str:
            if char.islower() and char in self.generators:
                idx = self.generators.index(char) + 1
                result.append(idx)
            elif char.isupper() and char.lower() in self.generators:
                idx = self.generators.index(char.lower()) + 1
                result.append(-idx)
            # Ignore characters not in the generator set
        
        # Perform free reduction
        reduced = []
        for g in result:
            if reduced and reduced[-1] == -g:
                reduced.pop()
            else:
                reduced.append(g)
        
        return reduced
    
    def __len__(self):
        """Return the length of the freely reduced word."""
        return len(self.word)
    
    def __eq__(self, other):
        """Check if two words are equal."""
        if not isinstance(other, FreeGroupWord):
            return False
        return self.word == other.word
    
    def __hash__(self):
        """Hash function for the word."""
        return hash(tuple(self.word))
    
    def __str__(self):
        """Convert the word back to a string representation."""
        result = []
        for g in self.word:
            if g > 0:
                result.append(self.generators[g-1])
            else:
                result.append(self.generators[-g-1].upper())
        return ''.join(result)
    
    def apply_automorphism(self, images):
        """
        Apply an automorphism to the word.
        
        Args:
            images: Dictionary mapping generator indices to their images
                   (lists of integers representing words)
        
        Returns:
            A new FreeGroupWord after applying the automorphism
        """
        # Apply the automorphism letter by letter
        new_word = []
        for g in self.word:
            if g > 0:
                new_word.extend(images[g])
            else:
                # For the inverse, we need the inverse of the image
                inverse_image = self._inverse_word(images[-g])
                new_word.extend(inverse_image)
        
        # Create a new word from the result
        result = FreeGroupWord("", self.generators)
        result.word = self._reduce_word(new_word)
        return result
    
    def _inverse_word(self, word):
        """Return the inverse of a word."""
        return [-g for g in reversed(word)]
    
    def _reduce_word(self, word):
        """Perform free reduction on a word."""
        reduced = []
        for g in word:
            if reduced and reduced[-1] == -g:
                reduced.pop()
            else:
                reduced.append(g)
        return reduced
    
    def cyclically_reduce(self):
        """Return the cyclically reduced form of the word."""
        if not self.word:
            return self
        
        # Keep removing paired elements from the beginning and end
        word = self.word.copy()
        while word and len(word) >= 2 and word[0] == -word[-1]:
            word = word[1:-1]
        
        result = FreeGroupWord("", self.generators)
        result.word = word
        return result


def apply_whitehead_automorphism(word, a, X):
    """
    Apply a Whitehead automorphism to a word.
    
    Args:
        word (FreeGroupWord): The word to transform
        a (int): The distinguished generator index (positive or negative)
        X (set): Set of generator indices (positive and negative)
                 that must include |a| and a
    
    Returns:
        A new FreeGroupWord after applying the automorphism
    """
    # Ensure a is in X
    if a not in X or -a not in X:
        raise ValueError("Both a and -a must be in X")
    
    n = len(word.generators)
    generator_indices = list(range(1, n+1))
    all_indices = generator_indices + [-i for i in generator_indices]
    
    # Create the automorphism mapping
    images = {}
    for g in generator_indices:
        if g == abs(a):
            # The distinguished generator maps to itself
            images[g] = [g]
        else:
            # Apply Whitehead automorphism rules
            g_in_X = g in X
            neg_g_in_X = -g in X
            
            if g_in_X and not neg_g_in_X:
                # g is in X but g^-1 is not
                if a > 0:
                    images[g] = [a, g]
                else:
                    images[g] = [g, -a]
            elif not g_in_X and neg_g_in_X:
                # g is not in X but g^-1 is
                if a > 0:
                    images[g] = [-a, g]
                else:
                    images[g] = [g, a]
            else:
                # Either both g and g^-1 are in X, or neither is
                images[g] = [g]
    
    return word.apply_automorphism(images)


def generate_whitehead_subsets(n, a):
    """
    Generate all possible subsets for Whitehead automorphisms.
    
    Args:
        n (int): Number of generators
        a (int): The distinguished generator index
    
    Returns:
        list: List of sets representing the possible X subsets
    """
    from itertools import combinations
    
    # Generate all generator indices and their inverses
    all_indices = list(range(1, n+1)) + list(range(-n, 0))
    
    # a and -a must be in X
    base_set = {abs(a), -abs(a)}
    
    # Generate all subsets containing a and -a
    result = []
    for r in range(len(all_indices) + 1):
        for subset in combinations(all_indices, r):
            current_set = base_set.copy()
            current_set.update(subset)
            if a in current_set and -a in current_set:
                result.append(current_set)
    
    return result


def find_minimal_elements(word):
    """
    Find all minimal length elements in the automorphism orbit of a word.
    
    Args:
        word (FreeGroupWord): The word to analyze
    
    Returns:
        set: Set of minimal length elements in the orbit
    """
    n = len(word.generators)
    current_min_length = len(word)
    current_min_elements = {word}
    queue = [word]
    visited = {word}
    
    while queue:
        current_word = queue.pop(0)
        
        # Try all possible Whitehead automorphisms
        for a_val in range(1, n+1):
            for a in [a_val, -a_val]:  # Try both the generator and its inverse
                # Generate all possible subsets X containing a
                for X in generate_whitehead_subsets(n, a):
                    try:
                        new_word = apply_whitehead_automorphism(current_word, a, X)
                        
                        if new_word not in visited:
                            visited.add(new_word)
                            length = len(new_word)
                            
                            if length < current_min_length:
                                # Found a shorter word, reset everything
                                current_min_length = length
                                current_min_elements = {new_word}
                                queue = [new_word]
                            elif length == current_min_length:
                                # Found another word of minimal length
                                current_min_elements.add(new_word)
                                queue.append(new_word)
                    except ValueError:
                        # Skip invalid Whitehead automorphisms
                        continue
    
    return current_min_elements


def is_automorphic(word1_str, word2_str, generators="abcd"):
    """
    Check if two words in a free group are automorphic.
    
    Args:
        word1_str (str): First word as a string (e.g., "abAc")
        word2_str (str): Second word as a string
        generators (str): String of available generators (default: "abcd")
    
    Returns:
        bool: True if the words are automorphic, False otherwise
    """
    # Create the free group words
    word1 = FreeGroupWord(word1_str, generators)
    word2 = FreeGroupWord(word2_str, generators)
    
    # If the words have different lengths after free reduction,
    # they can't be automorphic
    if len(word1) != len(word2):
        return False
    
    # For efficiency, use cyclically reduced forms
    word1 = word1.cyclically_reduce()
    word2 = word2.cyclically_reduce()
    
    # If the cyclically reduced forms have different lengths,
    # they can't be automorphic
    if len(word1) != len(word2):
        return False
    
    # For very simple cases, we can do a quick check
    if len(word1) <= 1:
        return len(word1) == len(word2)
    
    # Get minimal length elements in the automorphism orbit of word1
    min_elements1 = find_minimal_elements(word1)
    
    # Get minimal length elements in the automorphism orbit of word2
    min_elements2 = find_minimal_elements(word2)
    
    # Check if the minimal sets intersect
    for w1 in min_elements1:
        for w2 in min_elements2:
            if w1 == w2:
                return True
    
    return False


def main():

    result = is_automorphic(["a"], ["baB"], ["a","b"])
    print(result)


if __name__ == "__main__":
    main()