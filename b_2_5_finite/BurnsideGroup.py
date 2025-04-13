import time
import math

class BurnsideGroup:
    def __init__(self, generators, period):
        self.generators = generators
        self.period = period
        self.words = set(generators)
        self.relations = set()

    def to_aperiodic(self, word):
        while True:
            new_word = word
            for length in range(1, len(word) // self.period + 1):
                for i in range(len(word) - self.period * length + 1):
                    if word[i:i + length] == word[i + length:i + (self.period - 1) * length]\
                            == word[i + (self.period - 1) * length:i + self.period * length]:
                        new_word = word[:i] + word[i + self.period * length:]
                        break
                if new_word != word:
                    break
            if new_word == word:
                break
            word = new_word
            for relation in self.relations:
                word = word.replace(relation[0], relation[1])
        return word

    def sort_words(self, strings: list[str]) -> list[str]:
        return sorted(strings, key=lambda word: (len(word), word))

    def find_sim_in_table(self):
        strings = list(self.words)
        strings = self.sort_words(strings)
        n = len(strings)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.to_aperiodic(strings[i] + strings[j]))
            matrix.append(row)
        result = set()
        for row in matrix:
            value_to_columns = {}
            for idx, value in enumerate(row):
                if value in value_to_columns:
                    result.add(tuple(self.sort_words(list((strings[value_to_columns[value]], strings[idx])))))
                else:
                    value_to_columns[value] = idx
        self.relations.update(result)
        return True if result else False

    def generate_group(self):
        print(self.words)
        length = 0
        while True:
            new_words = set()
            for gen in self.generators:
                for word in self.words:
                    new_words.add(self.to_aperiodic(gen + word))
            self.words.update(new_words)
            while True:
                if self.find_sim_in_table():
                    for word in self.words:
                        new_word = self.to_aperiodic(word)
                        if new_word != word:
                            self.words.remove(word)
                            self.words.add(new_word)
                else:
                    break
            if length == len(self.words):
                break
            length = len(self.words)
            time.sleep(5)