import numpy as np
import pandas as pd
import logging
from . import utils as u

_logger = logging.getLogger('dimanalysis')

# Applied Dimensional Analysis and Modeling, pp. 165

def dimensional_matrix(*dvars):
    '''Returns the dimensional matrix formed by the given variables.'''
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

def ensure_nonsingular_A(dm, q, max_attempts=5):
    '''Ensures that submatrix A of the dimensional matrix is nonsingular.

    Nonsingularity of A is required as the inverse of A is used to determine
    exponent coefficients of the solution. The method implemented is aligned
    with "Applied Dimensional Analysis and Modeling" pp. 144-147 but differs
    in that randomized row-delete / col-swap attempts are executed. Thus,
    even if A could be made nonsingular, this method might fail. In case, try 
    to increase the number of attempts made.

    Params
    ------
    dm : pd.DataFrame
        Dimensional matrix (DxV) as returned by `dimensional_matrix`. We use dataframes to keep track of original row/column names.
    q : pd.DataFrame
        Dimensional matrix (Dx1) representing the target exponents. Passed, to keep ops on `dm` in sync with `q`.
    max_attempts: int, optional
        Number of randomized attempts to make A nonsingular.

    Returns
    -------
    dm' : pd.DataFrame
        Potentially reduced dimensional matrix (D'xV) to make A nonsingular
    q' : pd.DataFrame
        Potentially reduced target dimensional matrix (D'x1) required to make A
        nonsingular

    Raises
    ------
    ValueError
        If A could not be made nonsingular.
    '''
        
    def correct(dm, q):
        if not np.isclose(np.linalg.det(matrix_A(dm)), 0):
            return dm, q, True
        # A is singular
        n_d, n_v = dm.shape
        r_dm = np.linalg.matrix_rank(dm)
        delta = n_d - r_dm
        if delta > 0:
            # Number of dimensions exceeds rank -> no exchange could make 
            # rightmost det nonzero apply method 2
            row_ids = np.random.choice(n_d, size=delta, replace=False)
            dm = dm.drop(dm.index[row_ids])
            q = q.drop(q.index[row_ids])
            if not np.isclose(np.linalg.det(matrix_A(dm)), 0):
                return dm, q, True
        # A is still non-singular and delta = 0 -> randomly permute columns
        dm = dm.reindex(columns=np.random.permutation(dm.columns))
        return dm, q, not np.isclose(np.linalg.det(matrix_A(dm)), 0)

    # Remove all-zero rows that correspond to absent dimensions
    zero_row_mask = u.zero_rows(dm)
    dm = dm.drop(dm.index[zero_row_mask])
    q = q.drop(q.index[zero_row_mask])
    # Randomized attempts to fix singularity of A 
    for _ in range(max_attempts):
        dm_new, q, success = correct(dm)
        if success:
            return dm_new, q
    # All attempts exhausted, fail
    _logger.warn(
        'Matrix A is singular. All attempts to make it nonsingular failed.' \
        'However this method is non-deterministic. Try increasing the number' \
        'of attempts.')
    raise ValueError('Matrix A singular.')

def solve(*dvars, target=None):
    if target is None:
        target = dvars[0] / dvars[0] # unity, dimensionless products


    # Ensure that submatrix A is nonsingular
    dm = dimensional_matrix(*dvars)
    q = dimensional_matrix(target)  
    dm = ensure_nonsingular_A(dm, q)
    
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