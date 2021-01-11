import numpy as np
import pandas as pd
import logging
import itertools as it

from . import utils as u

_logger = logging.getLogger('dimensional_analysis')

# Applied Dimensional Analysis and Modeling, pp. 165

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
        if u.dimensionless(q):
            self.n_p = self.n_v - self.rank
        else:
            self.n_p = self.n_v - self.rank + 1
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

def matrix_E(A, B, dm_meta):
    '''Return extended matrix E consisting of blocks of A and B.'''
    A = np.asarray(A)
    B = np.asarray(B)
    rank, n_v = dm_meta.rank, dm_meta.n_v    
    Ainv = np.linalg.inv(A)
    return np.block([
        [np.eye(n_v-rank), np.zeros((n_v-rank, rank))],
        [-Ainv@B, Ainv]
    ])

def matrix_Z(qr, dm_meta):
    N,M = dm_meta.shape_e

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
    
    N,M = dm_meta.shape_q
    return np.block([[e],[np.tile(qr.reshape(-1,1), (1,M))]])

def ensure_nonsingular_A(dm, dm_meta):
    '''Ensures that submatrix A of the dimensional matrix is nonsingular.

    Nonsingularity of A is required as the inverse of A is used to determine
    exponent coefficients of the solution. The method implemented is aligned
    with "Applied Dimensional Analysis and Modeling" pp. 144-147.

    Params
    ------
    dm : DxV array
        Dimensional matrix (DxV) as returned by `dimensional_matrix`. 
    dm_meta : DimensionalSystemMeta
        Meta information about the system to solve for.

    Returns
    -------
    row_ids: array-like
        Row indices to delete
    col_perm: array-like
        Column permutation

    Raises
    ------
    ValueError
        If A could not be made nonsingular.
    '''
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
        dmr, _ = u.remove_rows(dm, row_ids)
        dmr, _ = u.permute_columns(dmr, col_ids)
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

def solve(dm, q=None):
    if not isinstance(dm, np.ndarray):
        dm = u.dimensional_matrix(dm)
    if q is None:
        q = np.zeros(dm.shape[0]) # unity, dimensionless products
    else:
        q = np.asarray(q)
    assert dm.ndim == 2, 'Invalid dimensional matrix dimensions.'
    assert dm.size > 0, 'Need at least one variable.'    
    assert len(q) == dm.shape[0], 'Target dimensions has incorrect number of components'

    dm_meta = DimensionalSystemMeta(dm, q)
    drow_ids, col_perm = ensure_nonsingular_A(dm, dm_meta)

    # All-zero rows of dm matrix correspond to non zero entries
    # in q. This makes no sense, as missing dimension in a system 
    # of variables cannot be restored.
    if any([q[i]!=0. for i in u.zero_rows(dm)]):        
        _logger.error( 
            'All-zero rows of dimensional matrix must correspond' \
            'to zero components in q.')
        raise ValueError('All-zero row r has to imply q[r]=0.')

    dmr, orow_ids = u.remove_rows(dm, drow_ids)
    dmr, inv_col_perm = u.permute_columns(dmr, col_perm)
    qr, _ = u.remove_rows(q, drow_ids) 
    # Recompute meta information on potentially reduced matrix
    dmr_meta  = DimensionalSystemMeta(dmr, qr)

    # Dimensional matrix (dmr) has more dims than vars, then possible have
    # a solution its rank must be rank <= vars - 1 by Theorem 7-6.
    if (u.dimensionless(qr) and 
        dmr_meta.n_d > dmr_meta.n_v and 
        dmr_meta.rank > dmr_meta.n_v - 1
        ):
        _logger.error( 
            'Rank of dimensional matrix must be <= number of vars - 1' \
            'by Theorem 7-6.'
        )
        raise ValueError('Rank must be <= number of vars - 1. See Theorem 7-6.')

    # If dmr is square it must be singular when number of non-zero
    # q components is zero. See Theorem 7-5 "Applied Dimensional 
    # Analysis and Modeling". 
    if dmr_meta.square and u.dimensionless(qr) and not dm_meta.singular:
        _logger.error(
            '(Reduced) Dimensional matrix must be singular when q is dimensionless' \
            'and number of variables equals number of dimensions.' \
            'See Theorem 7-5.'
        )
        raise ValueError('(Reduced) dimensional matrix not singular. See Theorem 7-5.')

    # Form E and Z
    A, B = matrix_A(dmr), matrix_B(dmr)
    E = matrix_E(A, B, dm_meta)
    Z = matrix_Z(qr, dm_meta)
    assert E.shape == (dm_meta.shape_E)    
    assert Z.shape == (dm_meta.shape_Z)

    # Form independent variable products
    P = E @ Z
    # Fix column reorder and return transpose, i.e solutions in rows
    return P.T[:, inv_col_perm]

    