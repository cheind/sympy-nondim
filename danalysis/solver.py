import itertools as it
from typing import Optional, List, Dict, Mapping, Union, Iterable
from collections import abc
from dataclasses import dataclass
import string
import logging

import numpy as np

from . import utils as u
from . import solver_utils as su
from . import sanity_checks as checks
from . import meta
from . import quantities as qt

_logger = logging.getLogger('danalysis')

@dataclass
class SolverOptions:
    remove_row_ids: Optional[List[int]] = None
    col_perm: Optional[List[int]] = None
    e: np.ndarray = None

def solve(dm, q, info=None, opts=None, strict=True):
    if info is None:
        info = meta.info(dm, q)
    if opts is None:
        opts = SolverOptions()
    dm = np.atleast_2d(dm)
    q = np.asarray(q)

    drow_ids, col_perm = su.ensure_nonsingular_A(dm, info, remove_row_ids=opts.remove_row_ids, col_perm=opts.col_perm)
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
    A, B = su.matrix_A(dmr, info), su.matrix_B(dmr, info)
    E = su.matrix_E(A, B, info)
    Z = su.matrix_Z(qr, info, e=opts.e)
    
    # Form independent variable products (in columns)
    P = E @ Z    
    # Revert column reorder and return transpose so that
    # indep. variable products are in rows.
    PT = P.T[:, inv_col_perm]

    # Check result dimension. 
    checks.result_dimension_match(PT, dm, q, strict)
    
    return PT


@dataclass
class Result:
    info: meta.LinearSystemMeta
    dimsys: qt.DimensionalSystem
    P: np.ndarray
    variables: Dict[str, qt.Q]

    @property
    def result_q(self) -> qt.Q:
        '''Dimension of each variable product.'''
        qs = self.variables.values()
        fd = u.variable_product(qs, self.P[0])
        return self.dimsys.q(fd)

    def __str__(self) -> str:        
        if len(self.P) == 0:
            return 'Found 0 solutions.'        

        names = self.variables.keys()
        finaldim = self.result_q
        vitems = [f'\t{n}:{v!r}' for n,v in self.variables.items()]
        vinner = ",\n".join(vitems)
        vset = f'\n{{\n{vinner}\n}}'

        m = (
            f'Found {len(self.P)} variable products of variables {vset}, each ' \
            f'of dimension {finaldim}:\n'
        )
        for i,p in enumerate(self.P):
            # Here we reuse fmt_dimensions as computing the product of variables
            # is the same as computing the derived dimension.
            s = qt._fmt_dimensions(p, names)
            m = m + f'\t{i+1}: [{s}] = {finaldim}\n'
        return m

class Solver:
    variables: Dict[str, qt.Q]
    q: qt.Q
    dm: np.ndarray
    info: meta.LinearSystemMeta

    def __init__(self, variables: Union[Mapping[str, qt.Q], Iterable[qt.Q]], q: qt.Q):
        if isinstance(variables, abc.Mapping):
            variables = dict(variables)
        elif isinstance(variables, abc.Iterable):
            variables = {n:v for n,v in zip(string.ascii_lowercase, variables)}
        else:
            raise ValueError('Variables needs to be mapping or sequence.')

        if (not all([isinstance(v, qt.Q) for v in variables.values()]) or 
                not isinstance(q, qt.Q)):
            raise ValueError('Variables and q need to be Q types.')
            
        self.variables = variables
        self.q = q
        self.dm = u.dimensional_matrix(self.variables.values())
        self.info = meta.info(self.dm, self.q)

    def solve(self, 
            select_values: Dict[str, Iterable[float]] = None) -> np.ndarray:
        opts = SolverOptions()
        if select_values is not None:
            if len(select_values) != self.info.n_free_variables:
                raise ValueError(
                    f'Number of variables in `select_values` does not '
                    f'match number of free variables {self.info.n_free_variables}'
                )            
            vn = list(self.variables.keys())
            fn = list(select_values.keys())
            invalid_vars = set(fn) - set(vn)
            if len(invalid_vars) != 0:
                raise ValueError(
                    f'Unknown variables selected {invalid_vars}'
                )
            # Move variables of free_values to the left via
            # col_perm field in SolverOptions
            leftids = [vn.index(n) for n in fn]
            allids = set(range(self.info.n_v))
            rightids = list(allids - set(leftids))
            opts.col_perm = leftids + rightids
            # Construct e matrix as Nfree x Nsol matrix
            opts.e = np.array(list(select_values.values()))

        P = solve(self.dm, self.q, self.info, opts=opts)
        return Result(self.info, self.q.dimsys, P, self.variables)
