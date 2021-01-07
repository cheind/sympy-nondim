import numpy as np
from .quantities import Quantity

# Applied Dimensional Analysis and Modeling, pp. 165

def dimensional_matrix(*dvars):
    '''Returns the dimensional matrix
    
    Params
    ------
    vars : array-like
        Quantities representing the system variables.

    Returns
    -------
    dm : DxV array
        Dimensional matrix of shape DxV
    '''

    vs = [np.asarray(v) for v in dvars]
    return np.stack(vs, -1)



class AugmentedMatrix:
    def __init__(self, m):
        r,c = m.shape
        self._m = np.empty((r+1,c+1), dtype=m.dtype)
        self._m[1:, 1:] = m
        self._m[0, 1:] = np.arange(c)
        self._m[1:, 0] = np.arange(r)

    @property
    def matrix(self):
        return self._m[1:, 1:]

    @property
    def indices(self):
        return self._m[1:, 0], self._m[0, 1:]

    def remove_rows(self, indices):
        r,c = self._m.shape
        mask = np.ones(r, dtype=bool)
        mask[indices+1] = 0
        self._m = self._m[mask]

    def swap_rows(self, i, j):
        self._m[[i+1,j+1]] = self._m[[j+1,i+1]]

    def swap_columns(self, i, j):
        self._m[:, [i+1,j+1]] = self._m[:, [j+1,i+1]]

    def zero_rows(self):
        return np.where(np.all(np.isclose(self.matrix, 0), -1))[0]


def dimensional_set(dm, q):
    assert isinstance(q, Quantity)

    dm = _remove_zero_rows(dm)
    nd, nv = dm.shape
    rdm = np.linalg.matrix_rank(dm)

