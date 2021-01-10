import numpy as np
import pandas as pd
import logging
import itertools as it
from . import utils as u

_logger = logging.getLogger('dimanalysis')

# Applied Dimensional Analysis and Modeling, pp. 165

def dimensional_matrix(*dvars):
    '''Returns the dimensional matrix formed by the given variables.'''
    data = {i:v.exponents for i,v in enumerate(dvars)}
    dm = pd.DataFrame(data, index=np.arange(len(data[0])))
    return dm

class DimensionalSystemMeta:
    def __init__(self, dm, q):
        self.dm = dm
        """Number of dimensions"""
        self.n_d = dm.shape[0]
        """Number of variables"""
        self.n_v = dm.shape[1]
        """Rank of dimensional matrix"""
        self.rank = np.linalg.matrix_rank(dm)
        """Number of possible independent variable products"""
        self.n_p = None
        if q.dimensionless:
            self.n_p = self.n_v - self.n_d
        else:
            self.n_p = self.n_v - self.n_d + 1
        """Shape of matrix A"""
        self.shape_A = (self.rank, self.rank)

    @property
    def square(self):
        return self.n_d == self.n_v

    @property
    def delta(self):
        return self.n_d - self.rank
    
    @property
    def singular(self):
        return not self.square or np.close(np.linalg.det(self.dm),0)

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

def nonsingular_A(dm, dm_meta):
    assert isinstance(dm_meta, DimensionalSystemMeta)
    zero_row_mask = np.all(np.isclose(dm, 0), -1)
    zero_row_ids = np.where(zero_row_mask)[0]
    n_z = len(zero_row_ids)

    # possible ways to remove rows to adjust for rank deficit 
    # (excluding zero rows)
    rowc = it.combinations(np.where(~zero_row_mask)[0], r=dm_meta.delta-n_z)
    # Possible ways to swap columns. Specific elements of permutation
    # that represent no-op swaps (in terms of singularity of A) are
    # filtered below
    cols_of_A = set(range(dm_meta.n_v-dm_meta.shape_A[0],dm_meta.n_v))
    colp = it.permutations(range(dm_meta.n_v))
    def valid_col_permutation(i,p):
        # Determine if this permutation is actually bringing a new 
        # column to A. i==0 is also allowed as it represented the
        # identity transform which we need to test anyways.
        # TODO consider Theorem 9-8 for a speed-up.
        s = set(p[dm_meta.shape_A[1]:])
        return i == 0 or len(cols_of_A ^ s) > 0
    colp = (p for i,p in enumerate(colp) if valid_col_permutation(i,p))

    for idx, (row_ids, col_ids) in enumerate(it.product(rowc, colp)):
        # Always delete all-zero rows as they represent dimensions
        # that are not represented in the variables.
        row_ids = list(row_ids) + zero_row_ids.tolist()
        dmr = dm.drop(dm.index[row_ids])
        dmr = dmr.reindex(columns=col_ids)
        if not np.isclose(np.linalg.det(matrix_A(dmr)), 0):
            _logger.debug(
                f'Deletable rows {row_ids}'\
                f', column order {col_ids}.'
            )
            return (row_ids, col_ids)

    # Failed
    _logger.error( 
        'Matrix A is singular.' \
        'All attempts to make it nonsingular failed.')
    raise ValueError('Matrix A singular.')



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