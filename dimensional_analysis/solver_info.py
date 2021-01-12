import string
import numpy as np
from . import utils as u

class SolverInfo:
    def __init__(self, variables, q):
        if isinstance(variables, dict):
            self.variables = list(variables.values())
            self.variable_names = list(variables.keys())
        else:
            self.variables = list(variables)
            self.variable_names = string.ascii_lowercase[:len(self.variables)]

        if len(variables) == 0:
            raise ValueError('Need at least one variable to continue')
        self.dm = u.dimensional_matrix(self.variables)

        '''Target dimensions'''
        self.q = None
        if q is None:
            self.q = np.zeros(self.dm.shape[0]) # unity, dimensionless products
        else:
            self.q = np.asarray(q)
        if len(self.q) != self.dm.shape[0]:
            raise ValueError('Target dimensionality does not match variable dimensionality')

        '''Number of dimensions'''
        self.n_d = self.dm.shape[0]
        '''Number of variables'''
        self.n_v = self.dm.shape[1]
        '''Rank of dimensional matrix'''
        self.rank = np.linalg.matrix_rank(self.dm)
        '''Number of possible independent variable products'''
        self.n_p = None
        self.dimensionless = u.dimensionless(self.q)
        if self.dimensionless:
            self.n_p = self.n_v - self.rank
        else:
            self.n_p = self.n_v - self.rank + 1
        '''Number of selectable components (i.e rows/dimensions)'''
        self.n_s = self.rank
        '''Shape of nonsingular matrix A'''
        self.shape_A = (self.rank, self.rank)
        '''Shape of matrix B to the left of A'''
        self.shape_B = (self.rank, self.n_v - self.rank)
        '''Shape of matrix e containing the freely selectable exponents.'''
        self.shape_e = (self.n_v - self.rank, self.n_p)
        '''Shape of the matrix q that repeats the selectable dimensions of input `q` accross cols'''
        self.shape_q = (self.rank, self.n_p)
        '''Shape of matrix Z which represents (e,q) stacked along rows.'''
        self.shape_Z = (self.n_v, self.n_p)
        '''Shape of matrix E which contains blocks of I,0,A,B'''
        self.shape_E = (self.n_v, self.n_v)

    @property
    def square(self):
        return self.n_d == self.n_v

    @property
    def delta(self):
        return self.n_d - self.rank
    
    @property
    def singular(self):
        return not self.square or np.close(np.linalg.det(self.dm),0)

def solver_info(variables, q):
    return SolverInfo(variables, q)