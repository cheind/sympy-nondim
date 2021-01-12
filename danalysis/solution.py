from . import utils as u
from .quantities import _fmt_dimensions

class Solution:
    def __init__(self, info, P):
        self.P = P
        self.info = info

    def __repr__(self):
        return f'Solution<{self.P}>'

    def __str__(self):        
        if len(self.P) == 0:
            return 'Found 0 solutions.'        

        Vcls = self.info.variables[0].__class__    
        finaldim = Vcls(u.variable_product(self.info.variables, self.P[0]))
        m = f'Found {len(self.P)} variable products of dimension {finaldim}:\n'
        for i,p in enumerate(self.P):
            # Here we reuse fmt_dimensions as computing the product of variables
            # is the same as computing the derived dimension.
            s = _fmt_dimensions(p, self.info.variable_names)
            m = m + f'{i:4}: {s}\n'
        return m