from burau_representation.classes.generator_element import GeneratorElement


class Generators:
    def __init__(self, mod=None):
        self.mod = mod

        self.A = GeneratorElement('A', mod)
        self.B = GeneratorElement('B', mod)
        self.a = GeneratorElement('a', mod)
        self.b = GeneratorElement('b', mod)

    def __getitem__(self, key):
        if key not in {'A', 'B', 'a', 'b'}:
            raise KeyError(f"Invalid generator key: {key}")
        return getattr(self, key)