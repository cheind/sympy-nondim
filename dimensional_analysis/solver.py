import numpy as np
import itertools as it
import logging

from . import utils as u
from . import sanity_checks as checks
from .solver_info import solver_info
from .solution import Solution

_logger = logging.getLogger('dimensional_analysis')

# Applied Dimensional Analysis and Modeling, pp. 165

def _matrix_A(dm, info):
    '''Returns submatrix A of dimensional matrix'''
    n_d = dm.shape[0]
    A = dm[:, -n_d:]
    assert A.shape == info.shape_A
    return A

def _matrix_B(dm, info):
    '''Returns submatrix B of dimensional matrix'''
    n_d = dm.shape[0]
    B = dm[:, :dm.shape[1]-dm.shape[0]]   
    assert B.shape == info.shape_B
    return B

def _matrix_E(A, B, info):
    '''Return extended matrix E consisting of blocks of A and B.'''
    A = np.asarray(A)
    B = np.asarray(B)
    rank, n_v = info.rank, info.n_v    
    Ainv = np.linalg.inv(A)
    E = np.block([
        [np.eye(n_v-rank), np.zeros((n_v-rank, rank))],
        [-Ainv@B, Ainv]
    ])
    assert E.shape == (info.shape_E)    
    return E

def _matrix_Z(qr, info):
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
        if N*M > 0:
            e[0, -1] = 1
            e[1, -1] = 1
    
    N,M = info.shape_q
    Z = np.block([[e],[np.tile(qr.reshape(-1,1), (1,M))]])
    assert Z.shape == (info.shape_Z)
    return Z


def _row_removal_generator(dm, info, keep_rows=None):
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

def _column_permutation_generator(info):
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


def _ensure_nonsingular_A(dm, info, keep_rows=None):
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
    row_gen = _row_removal_generator(dm, info, keep_rows=keep_rows)
    col_gen = _column_permutation_generator(info)

    for idx, (row_ids, col_ids) in enumerate(it.product(row_gen, col_gen)):
        # Always delete all-zero rows as they represent dimensions
        # that are not represented in the variables.
        dmr, _ = u.remove_rows(dm, row_ids)
        dmr, _ = u.permute_columns(dmr, col_ids)        
        if not np.isclose(np.linalg.det(_matrix_A(dmr, info)), 0):
            _logger.debug(
                f'Removing rows {row_ids}'\
                f', new column order {col_ids}.'
            )
            return (row_ids, col_ids)

    # Failed
    _logger.error( 
        'Matrix A is singular.' \
        'All attempts to make it nonsingular failed.')
    raise ValueError('Matrix A singular')   

def solve(variables, q=None, keep_rows=None):
    info = solver_info(variables, q)
    dm = info.dm
    q = info.q

    drow_ids, col_perm = _ensure_nonsingular_A(dm, info, keep_rows=keep_rows)
    checks.assert_zero_q_when_all_zero_rows(dm, q)  
    if info.n_s < info.n_d:
        _logger.info((
            f'Number of selectable components is less than number dimensions. '
            f'Values of non-selectable components {drow_ids} are computed '
            f'may differ from values in q, which may be unexpected.'
        ))

    dmr, orow_ids = u.remove_rows(dm, drow_ids)
    dmr, inv_col_perm = u.permute_columns(dmr, col_perm)
    qr, _ = u.remove_rows(q, drow_ids) 
    # Recompute meta information on potentially reduced matrix

    checks.assert_no_rank_deficit(dmr, qr, info.rank)
    checks.assert_square_singular(dmr, qr)

    # Form E and Z
    A, B = _matrix_A(dmr, info), _matrix_B(dmr, info)
    E = _matrix_E(A, B, info)
    Z = _matrix_Z(qr, info)
    
    # Form independent variable products (in columns)
    P = E @ Z    
    # Revert column reorder and return transpose so that
    # indep. variable products are in rows.
    return Solution(info, P.T[:, inv_col_perm])

    