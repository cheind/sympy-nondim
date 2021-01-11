import numpy as np
import pandas as pd
import logging
import itertools as it

from . import utils as u
from . import sanity_checks as checks

_logger = logging.getLogger('dimensional_analysis')

# Applied Dimensional Analysis and Modeling, pp. 165

class SolverInfo:
    def __init__(self, dm, dimensionless=True):
        self.dm = dm
        """Number of dimensions"""
        self.n_d = dm.shape[0]
        """Number of variables"""
        self.n_v = dm.shape[1]
        """Rank of dimensional matrix"""
        self.rank = np.linalg.matrix_rank(dm)
        """Number of possible independent variable products"""
        self.n_p = None
        self.dimensionless = dimensionless
        if self.dimensionless:
            self.n_p = self.n_v - self.rank
        else:
            self.n_p = self.n_v - self.rank + 1
        """Number of selectable components (i.e rows/dimensions)"""
        self.n_s = self.rank
        """Shape of nonsingular matrix A"""
        self.shape_A = (self.rank, self.rank)
        """Shape of matrix B to the left of A"""
        self.shape_B = (self.rank, self.n_v - self.rank)
        """Shape of matrix e containing the freely selectable exponents."""
        self.shape_e = (self.n_v - self.rank, self.n_p)
        """Shape of the matrix q that repeats the selectable dimensions of input `q` accross cols"""
        self.shape_q = (self.rank, self.n_p)
        """Shape of matrix Z which represents (e,q) stacked along rows."""
        self.shape_Z = (self.n_v, self.n_p)
        """Shape of matrix E which contains blocks of I,0,A,B"""
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

def solver_info(dm, q):
    dimensionless = u.dimensionless(q)
    return SolverInfo(dm, dimensionless=dimensionless)

def matrix_A(dm):
    '''Returns submatrix A of dimensional matrix'''
    n_d = dm.shape[0]
    A = dm[:, -n_d:]
    return A

def matrix_B(dm):
    '''Returns submatrix B of dimensional matrix'''
    n_d = dm.shape[0]
    B = dm[:, :dm.shape[1]-dm.shape[0]]
    return B

def matrix_E(A, B, info):
    '''Return extended matrix E consisting of blocks of A and B.'''
    A = np.asarray(A)
    B = np.asarray(B)
    rank, n_v = info.rank, info.n_v    
    Ainv = np.linalg.inv(A)
    return np.block([
        [np.eye(n_v-rank), np.zeros((n_v-rank, rank))],
        [-Ainv@B, Ainv]
    ])

def matrix_Z(qr, info):
    N,M = info.shape_e

    if N==M:
        # Square e happens when dimensionless q is used, since then
        # N_V - Rdm (rows) = N_p (cols)
        e = np.eye(N)
    else:
        # Non-square N_V - Rdm + 1 (rows) = N_p (cols)
        # Ensure that columns are independet. We freely set the
        # last column to be [1,1,0...]
        e = np.zeros((N,M))
        e[:N, :N] = np.eye(N)
        e[:, -1] = np.zeros(N)
        e[0, -1] = 1
        e[1, -1] = 1
    
    N,M = info.shape_q
    return np.block([[e],[np.tile(qr.reshape(-1,1), (1,M))]])


def row_removal_generator(dm, info, keep_rows=None):
    if keep_rows is None:
        keep_rows = []

    zero_row_mask = np.all(np.isclose(dm, 0), -1)
    zero_row_ids = np.where(zero_row_mask)[0]
    n_z = len(zero_row_ids)
    keep_set = set(keep_rows)

    # Possible ways to remove rows to adjust for rank deficit 
    # (excluding zero rows). We also do not include any row i 
    # for which q_i != 0
    rowc = it.combinations(
        range(info.n_d),
        r=info.delta
    )
    
    def priority(r):
        n_keep = len(set(r) & keep_set)
        if n_keep > 0:
            return n_keep
        
        n_zeros = zero_row_mask[r].sum()
        if n_zeros > 0:
            return -n_zeros

        return 0

    return iter(sorted(rowc, key=priority))

def column_permutation_generator(info):
    # Possible ways to swap columns. Specific elements of permutation
    # that represent no-op swaps (in terms of singularity of A) are
    # filtered below
    cols_of_A = set(range(info.n_v-info.shape_A[0],info.n_v))
    colp = it.permutations(range(info.n_v))
    
    def effective_permutation(i,p):
        # Determine if this permutation is actually bringing a new 
        # column to A. i==0 is also allowed as it represented the
        # identity transform which we need to test anyways.
        # TODO consider Theorem 9-8 for a speed-up.
        s = set(p[info.shape_A[1]:])
        return i == 0 or len(cols_of_A ^ s) > 0
    gen = (p for i,p in enumerate(colp) if effective_permutation(i,p))
    yield from gen


def ensure_nonsingular_A(dm, info, keep_rows=None):
    '''Ensures that submatrix A of the dimensional matrix is nonsingular.

    Nonsingularity of A is required as the inverse of A is used to determine
    exponent coefficients of the solution. The method implemented is aligned
    with "Applied Dimensional Analysis and Modeling" pp. 144-147.

    Params
    ------
    dm : DxV array
        Dimensional matrix (DxV) as returned by `dimensional_matrix`. 
    info : SolverInfo
        Meta information about the system to solve for.
    keep_rows : array-like, optional
        Component (i.e row) indices that should not be removed if
        possible to make A nonsingular.

    Returns
    -------
    row_ids: array-like
        Row indices to drop
    col_perm: array-like
        Column permutation

    Raises
    ------
    ValueError
        If A could not be made nonsingular.
    '''
    assert isinstance(info, SolverInfo)

    row_gen = row_removal_generator(dm, info, keep_rows=keep_rows)
    col_gen = column_permutation_generator(info)

    for idx, (row_ids, col_ids) in enumerate(it.product(row_gen, col_gen)):
        # Always delete all-zero rows as they represent dimensions
        # that are not represented in the variables.
        dmr, _ = u.remove_rows(dm, row_ids)
        dmr, _ = u.permute_columns(dmr, col_ids)        
        if not np.isclose(np.linalg.det(matrix_A(dmr)), 0):
            _logger.debug(
                f'Removing rows {row_ids}'\
                f', new column order {col_ids}.'
            )
            return (row_ids, col_ids)

    # Failed
    _logger.error( 
        'Matrix A is singular.' \
        'All attempts to make it nonsingular failed.')
    raise ValueError('Matrix A singular.')

def solve(dm, q=None, keep_rows=None):
    if not isinstance(dm, np.ndarray):
        dm = u.dimensional_matrix(dm)
    if q is None:
        q = np.zeros(dm.shape[0]) # unity, dimensionless products
    else:
        q = np.asarray(q)
    assert dm.ndim == 2, 'Invalid dimensional matrix dimensions.'
    assert dm.size > 0, 'Need at least one variable.'    
    assert len(q) == dm.shape[0], 'Target dimensions has incorrect number of components'

    info = solver_info(dm, q)
    drow_ids, col_perm = ensure_nonsingular_A(dm, info, keep_rows=keep_rows)

    checks.assert_zero_q_when_all_zero_rows(dm, q)

    dmr, orow_ids = u.remove_rows(dm, drow_ids)
    dmr, inv_col_perm = u.permute_columns(dmr, col_perm)
    qr, _ = u.remove_rows(q, drow_ids) 
    # Recompute meta information on potentially reduced matrix
    dmr_meta  = solver_info(dmr, qr)

    checks.assert_no_rank_deficit(dmr_meta, qr)
    checks.assert_square_singular(dmr_meta, qr)

    # Form E and Z
    A, B = matrix_A(dmr), matrix_B(dmr)
    E = matrix_E(A, B, info)
    Z = matrix_Z(qr, info)
    assert E.shape == (info.shape_E)    
    assert Z.shape == (info.shape_Z)

    # Form independent variable products
    P = E @ Z
    # Fix column reorder and return transpose, i.e solutions in rows
    return P.T[:, inv_col_perm]

    