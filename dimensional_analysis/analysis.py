import numpy as np
from .quantities import Quantity

# Applied Dimensional Analysis and Modeling, pp. 165

def dimensional_matrix(*dvars):
    '''Returns the dimensional matrix from the given variables.
    
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

def dimensional_set(dm, q):
    assert isinstance(q, Quantity)

    dm = _remove_zero_rows(dm)
    nd, nv = dm.shape
    rdm = np.linalg.matrix_rank(dm)

