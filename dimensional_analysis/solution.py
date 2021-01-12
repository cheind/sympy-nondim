from . import utils as u

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
        m = f'Found {len(self.P)} variable products of dim. {finaldim}.\n'
        for i,p in enumerate(self.P):
            ve = [f'{v}**{e}' for v,e 
            in zip(self.info.variable_names, p) if e!=0]
            m = m + f'{i}: {"*".join(ve)}\n'
        return m