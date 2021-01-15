import numpy as np
import itertools as it
import logging

from . import sanity_checks as checks
from . import utils as u

_logger = logging.getLogger('danalysis')

def matrix_A(dm, info):
    '''Returns submatrix A of dimensional matrix'''
    N,M = info.shape_A
    A = dm[:N, -M:]
    return A

def matrix_B(dm, info):
    '''Returns submatrix B of dimensional matrix'''
    N,M = info.shape_B
    B = dm[:N, :M]   
    return B

def matrix_E(A, B, info):
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

def matrix_Z(qr, info, e=None):
    N,M = info.shape_e

    if e is not None:
        e = np.asarray(e)
        if e.shape[0] != N:
            checks._fail(
                f'e-matrix needs to have {N} rows.',
                critical=True
            )
    elif N==M:
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
        if N > 1:
            e[0, -1] = 1
            e[1, -1] = 1
    

    return np.block([
        [e],
        [np.tile(qr.reshape(-1,1), (1,e.shape[1]))]
    ])


def row_removal_generator(dm, info):
    # Possible ways to remove rows to adjust for rank deficit 
    # (excluding zero rows). We also do not include any row i 
    # for which q_i != 0
    rowc = it.combinations(
        range(info.n_d),
        r=info.delta
    )

    zero_row_mask = np.all(np.isclose(dm, 0), -1)    
    def priority(r):
        n_zeros = zero_row_mask[list(r)].sum()
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

def remove_rows(dm, info, remove_row_ids=None):
    def equal_ranks(dmr):
        return np.linalg.matrix_rank(dmr) == info.rank

    if remove_row_ids is not None:
        r = np.asarray(remove_row_ids)
        if len(r) != info.delta:
            checks._fail(
                (
                    f'Number of rows to delete {len(r)} does not '
                    f'match number of required rows {info.delta}.'
                ),
                critical=True
            )
        if np.any(r < 0 or r > info.n_d-1):
            checks._fail(
                f'Row indices out of bounds.',
                critical=True
            )
        m, _ = u.remove_rows(dm, r)
        if not equal_ranks(m):
            checks._fail(
                (
                    f'Preferred row deletion failed, because it leads '
                    f'to a rank < original rank.'
                ),
                critical=True
            )
        return list(r)

    row_gen = row_removal_generator(dm, info)    
    for r in row_gen:
        m, _ = u.remove_rows(dm, r)
        if equal_ranks(m):
            return list(r)
    
    # Should not happen
    checks._fail('Failed remove dimensional matrix rows.', critical=True)

def permute_cols(dmr, info, col_perm=None):
    def notsingular(dmr):
        return not np.isclose(np.linalg.det(matrix_A(dmr, info)), 0)

    if col_perm is not None:
        c = list(col_perm)
        if len(c) != info.n_v or len(set(c) & set(range(info.n_v))) != info.n_v:
            checks._fail(
                f'Not a valid column permutation {c}',
                critical=True
            )
        m, _ = u.permute_columns(dmr, c)
        if not notsingular(m):
            checks._fail(
                f'Preferred column permution yields singular A matrix.',
                critical=True
            )
        return c

    col_gen = column_permutation_generator(info)
    for c in col_gen:
        m, _ = u.permute_columns(dmr, c)
        if notsingular(m):
            return list(c)

    # Should not happen
    checks._fail('Failed to find nonsingular matrix A.', critical=True)

def ensure_nonsingular_A(dm, info, remove_row_ids=None, col_perm=None):
    '''Ensures that submatrix A of the dimensional matrix is nonsingular.

    Nonsingularity of A is required as the inverse of A is used to determine
    exponent coefficients of the solution. The method implemented is aligned
    with "Applied Dimensional Analysis and Modeling" pp. 144-147.

    Params
    ------
    dm : DxV array
        Dimensional matrix (DxV) as returned by `dimensional_matrix`. 
    info : LinearSystemInfo
        Meta information about the system to solve for.

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

    row_ids = remove_rows(dm, info, remove_row_ids=remove_row_ids)
    dmr, _ = u.remove_rows(dm, row_ids)
    col_perm = permute_cols(dmr, info, col_perm=col_perm)
    _logger.debug((
        f'Removing rows {row_ids}'
        f', new column order {col_perm}.'
    ))
    return (row_ids, col_perm)
