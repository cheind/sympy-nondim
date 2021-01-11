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
    '''Ensures that submatrix A of the dimensional matrix is nonsingular.

    Nonsingularity of A is required as the inverse of A is used to determine
    exponent coefficients of the solution. The method implemented is aligned
    with "Applied Dimensional Analysis and Modeling" pp. 144-147.

    Params
    ------
    dm : pd.DataFrame
        Dimensional matrix (DxV) as returned by `dimensional_matrix`. We use dataframes to keep track of original row/column names.
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

def solve(*dvars, q=None):
    if q is None:
        q = dvars[0] / dvars[0] # unity, dimensionless products

    dm = dimensional_matrix(*dvars)
    qdm = dimensional_matrix(q)
    dm_meta = DimensionalSystemMeta(dm, q)
    row_ids, col_perm = nonsingular_A(dm, dm_meta)

    # All-zero rows of dm matrix correspond to non zero entries
    # in q. This makes no sense, as missing dimension in a system 
    # of variables cannot be restored.
    if any([q[i]!=0. for i in u.zero_rows(dm)]):        
        _logger.error( 
            'All-zero rows of dimensional matrix must correspond' \
            'to zero components in q.')
        raise ValueError('All-zero row r has to imply q[r]=0.')

    # Dimensional matrix (dm) has more dims than vars, then possible have
    # a solution its rank must be rank <= vars - 1 by Theorem 7-6.
    # TODO: perform this test on the reduced dm.
    if (q.dimensionless and 
        dm_meta.n_d > dm_meta.n_v and 
        dm_meta.rank > dm_meta.n_v - 1
        ):
        _logger.error( 
            'Rank of dimensional matrix must be <= number of vars - 1' \
            'by Theorem 7-6.'
        )
        raise ValueError('Rank must be <= number of vars - 1. See Theorem 7-6.')

    # If dm is square it must be singular when number of non-zero
    # q components is zero. See Theorem 7-5 "Applied Dimensional 
    # Analysis and Modeling". 
    # TODO: perform this test on the reduced dm.
    if dm_meta.square and q.dimensionless and not dm_meta.singular:
        _logger.error(
            'Dimensional matrix must be singular when q is dimensionless' \
            'and number of variables equals number of dimensions.' \
            'See Theorem 7-5.'
        )
        raise ValueError('Dimensional matrix not singular. See Theorem 7-5.')

    dmr = dm.drop(dm.index[row_ids])
    dmr = dmr.reindex(columns=col_perm)
    qdmr = qdm.drop(dm.index[row_ids])
