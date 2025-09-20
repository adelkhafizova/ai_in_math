from burau_package.classes.generator_element import GeneratorElement


class Generators:
    def __init__(self, mod=None):
        self.mod = mod

        self.A = GeneratorElement('A', mod).matrix
        self.B = GeneratorElement('B', mod).matrix
        self.a = GeneratorElement('a', mod).matrix
        self.b = GeneratorElement('b', mod).matrix

    def __getitem__(self, key):
        if key not in {'A', 'B', 'a', 'b'}:
            raise KeyError(f"Invalid generator key: {key}")
        return getattr(self, key)