import numpy as np
import itertools as it
from typing import Optional, List
from dataclasses import dataclass
import logging

from . import utils as u
from . import sanity_checks as checks

_logger = logging.getLogger('danalysis')

class SolverInfo:
    def __init__(self, dm, q):
        dm = np.atleast_2d(dm)
        q = np.asarray(q)

        if dm.size == 0:
            raise ValueError('Need at least one variable/dimension.')
        if len(q) != dm.shape[0]:
            raise ValueError(
                'Target dimensionality does not match ' \
                'variable dimensionality')
            
        '''Number of dimensions'''
        self.n_d = dm.shape[0]
        '''Number of variables'''
        self.n_v = dm.shape[1]
        '''Rank of dimensional matrix'''
        self.rank = np.linalg.matrix_rank(dm)
        '''Number of possible independent variable products'''
        self.n_p = None
        self.dimensionless = u.dimensionless(q)
        if self.dimensionless:
            self.n_p = self.n_v - self.rank
        else:
            self.n_p = self.n_v - self.rank + 1
        '''Number of selectable components (i.e rows/dimensions)'''
        self.n_s = self.rank
        '''Shape of nonsingular matrix A'''
        self.shape_A = (self.rank, self.rank)
        '''Shape of matrix B to the left of A'''
        self.shape_B = (self.rank, self.n_v - self.rank)
        '''Shape of matrix e containing the freely selectable exponents.'''
        self.shape_e = (self.n_v - self.rank, self.n_p)
        '''Shape of the matrix q that repeats the selectable dimensions of input `q` accross cols'''
        self.shape_q = (self.rank, self.n_p)
        '''Shape of matrix Z which represents (e,q) stacked along rows.'''
        self.shape_Z = (self.n_v, self.n_p)
        '''Shape of matrix E which contains blocks of I,0,A,B'''
        self.shape_E = (self.n_v, self.n_v)

    @property
    def square(self) -> bool:
        return self.n_d == self.n_v

    @property
    def delta(self) -> int:
        return self.n_d - self.rank

    @property
    def n_free_variables(self) -> int:
        return self.n_v - self.rank

    @property
    def n_excess_rows(self) -> int:
        return self.delta
    
    @property
    def singular(self) -> bool:
        return not self.square or np.close(np.linalg.det(self.dm),0)

def solver_info(dm: np.ndarray, q: np.ndarray) -> SolverInfo:
    return SolverInfo(dm, q)

@dataclass
class SolverOptions:
    remove_row_ids: Optional[List[int]] = None
    col_perm: Optional[List[int]] = None
    e: np.ndarray = None


def _matrix_A(dm, info):
    '''Returns submatrix A of dimensional matrix'''
    N,M = info.shape_A
    A = dm[:N, -M:]
    return A

def _matrix_B(dm, info):
    '''Returns submatrix B of dimensional matrix'''
    N,M = info.shape_B
    B = dm[:N, :M]   
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

def _matrix_Z(qr, info, opts):
    N,M = info.shape_e

    if opts.e is not None:
        e = np.asarray(opts.e)
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


def _row_removal_generator(dm, info):
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

def _remove_rows(dm, info, opts):
    def equal_ranks(dmr):
        return np.linalg.matrix_rank(dmr) == info.rank

    if opts.remove_row_ids is not None:
        r = np.asarray(opts.remove_row_ids)
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

    row_gen = _row_removal_generator(dm, info)    
    for r in row_gen:
        m, _ = u.remove_rows(dm, r)
        if equal_ranks(m):
            return list(r)
    
    # Should not happen
    checks._fail('Failed remove dimensional matrix rows.', critical=True)

def _permute_cols(dmr, info, opts):
    def notsingular(dmr):
        return not np.isclose(np.linalg.det(_matrix_A(dmr, info)), 0)

    if opts.col_perm is not None:
        c = list(opts.col_perm)
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

    col_gen = _column_permutation_generator(info)
    for c in col_gen:
        m, _ = u.permute_columns(dmr, c)
        if notsingular(m):
            return list(c)

    # Should not happen
    checks._fail('Failed to find nonsingular matrix A.', critical=True)

def _ensure_nonsingular_A(dm, info, opts):
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
    opts : SolverOptions
        Advanced solver options

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

    row_ids = _remove_rows(dm, info, opts)
    dmr, _ = u.remove_rows(dm, row_ids)
    col_perm = _permute_cols(dmr, info, opts)
    _logger.debug((
        f'Removing rows {row_ids}'
        f', new column order {col_perm}.'
    ))
    return (row_ids, col_perm)

def solve(dm, q, info=None, opts=None, strict=True):
    if info is None:
        info = solver_info(dm, q)
    if opts is None:
        opts = SolverOptions()
    dm = np.atleast_2d(dm)
    q = np.asarray(q)

    drow_ids, col_perm = _ensure_nonsingular_A(dm, info, opts)
    checks.assert_zero_q_when_all_zero_rows(dm, q)  
    if info.n_s < info.n_d:
        _logger.info((
            f'Number of selectable dimensions (rows) is less than number '
            f'dimensions. Values of non-selectable dimensions {drow_ids} '
            f'are computed may differ target dimensionality.'
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
    Z = _matrix_Z(qr, info, opts)
    
    # Form independent variable products (in columns)
    P = E @ Z    
    # Revert column reorder and return transpose so that
    # indep. variable products are in rows.
    PT = P.T[:, inv_col_perm]

    # Check result dimension. 
    checks.result_dimension_match(PT, dm, q, strict)
    
    return PT
    