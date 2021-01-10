import numpy as np
import pandas as pd
import logging
from . import utils as u

_logger = logging.getLogger('dimanalysis')

# Applied Dimensional Analysis and Modeling, pp. 165

def dimensional_matrix(*dvars):
    '''Returns the dimensional matrix form the given variables.'''
    data = {i:v.exponents for i,v in enumerate(dvars)}
    dm = pd.DataFrame(data, index=np.arange(len(data[0])))
    return dm

def matrix_A(dm):
    '''Returns submatrix A of dimensional matrix'''
    dm = np.asarray(dm)
    n_d = dm.shape[0]
    A = dm[:, -n_d:]
    return A

def matrix_B(dm):
    '''Returns submatrix B of dimensional matrix'''
    dm = np.asarray(dm)
    n_d = dm.shape[0]
    B = dm[:, :n_d-1]
    return B

def matrix_E(A, B):
    '''Return extended matrix E consisting of blocks of A and B.'''
    A = np.asarray(A)
    B = np.asarray(B)
    n_d, n_v = A.shape[0], A.shape[1] + B.shape[1]
    Ainv = np.linalg.inv(A)
    return np.block([
        [np.eye(n_v-n_d), np.zeros((n_v-n_d, n_d))],
        [-Ainv@B, Ainv]
    ])

def ensure_nonsingular_A(dm, max_attempts=5):
    # Applied Dimensional Analysis and Modeling, pp. 144-147
    # Even dm can be made non-singular this method might not find a solution because of its monte-carlo nature.
        
    def correct(dm):
        if not np.isclose(np.linalg.det(matrix_A(dm)), 0):
            return dm, True
        # A is singular
        n_d, n_v = dm.shape
        r_dm = np.linalg.matrix_rank(dm)
        delta = n_d - r_dm
        if delta > 0:
            # Number of dimensions exceeds rank -> 
            # no exchange could make rightmost det nonzero
            # apply method 2
            row_ids = np.random.choice(n_d, size=delta, replace=False)
            dm = dm.drop(dm.index[row_ids])
            if not np.isclose(np.linalg.det(matrix_A(dm)), 0):
                return dm, True
        # A is still non-singular and delta = 0 ->
        # randomly permute columns
        dm = dm.reindex(columns=np.random.permutation(dm.columns))
        return dm, not np.isclose(np.linalg.det(matrix_A(dm)), 0)

    # Remove all-zero rows that correspond to absent dimensions
    zero_row_mask = u.zero_rows(dm)
    dm = dm.drop(dm.index[zero_row_mask])
    
    for _ in range(max_attempts):
        dm_new, success = correct(dm)
        if success:
            return dm_new
    # All attempts exhausted, fail
    raise ValueError('A singular')

def solve(*dvars, q=None):
    if q is None:
        q = dvars[0] / dvars[0] # dimensionless
    # Ensure that submatrix A is nonsingular
    dm = dimensional_matrix(*dvars)    
    dm = ensure_nonsingular_A(dm)
    
    # how to handle q_i!=0 but i-th dimension deleted?

    print(dm)

    pass

"""
def solve(dvars, target):
    dm = dimensional_matrix(*dvars)
    # Remove all-zero rows that correspond to unused dimensions
    utils.zero_rows(dm)


def dimensional_set(dm, q):
    assert isinstance(q, Quantity)

    dm = _remove_zero_rows(dm)
    nd, nv = dm.shape
    rdm = np.linalg.matrix_rank(dm)

# A singular? See pp. 144!
"""