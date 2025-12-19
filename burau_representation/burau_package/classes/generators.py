from burau_package.classes.generator_element import GeneratorElement


class Generators:
    def __init__(self, mod=None):
        self.mod = mod

        self.A = GeneratorElement('A', mod).matrix
        self.B = GeneratorElement('B', mod).matrix
        self.a = GeneratorElement('a', mod).matrix
        self.b = GeneratorElement('b', mod).matrix
        self.T = GeneratorElement('T', mod).matrix
        self.t = GeneratorElement('t', mod).matrix
        self.s0 = GeneratorElement('s_0', mod).matrix
        self.s1 = GeneratorElement('s_1', mod).matrix
        self.s2 = GeneratorElement('s_2', mod).matrix

    def __getitem__(self, key):
        if key not in {'A', 'B', 'a', 'b','T','t','s_0','s_1','s_2'}:
            raise KeyError(f"Invalid generator key: {key}")
        return getattr(self, key)