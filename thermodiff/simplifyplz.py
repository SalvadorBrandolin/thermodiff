from thermodiff.diffplz import DiffPlz


class SimplifyPlz:
    def __init__(self, diffplz: DiffPlz):
        self.name = diffplz.name
        self.expression = diffplz.expression
        self.internal_functions = diffplz.internal_functions
        self.indexes = diffplz.indexes
        
        
        