from collections import OrderedDict
from typing import Iterable, Any, Union
import numpy as np
import logging
import copy

from .quantities import DimensionalSystem, Q, _fmt_dimensions
from . import utils as u
from . import solver as slv

_logger = logging.getLogger('danalysis')

class Result:
    def __init__(
        self, 
        info: slv.SolverInfo, 
        dimsys: DimensionalSystem,
        P: np.ndarray, 
        vs: OrderedDict
    ):
        self.info = info
        self.dimsys = dimsys
        self.P = P
        self.variables = vs

    def __repr__(self):
        return f'Result<{self.P}>'

    @property
    def result_q(self) -> Q:
        '''Dimension of each variable product.'''
        qs = self.variables.values()
        fd = u.variable_product(qs, self.P[0])
        return self.dimsys.q(fd)

    def __str__(self):        
        if len(self.P) == 0:
            return 'Found 0 solutions.'        

        names = self.variables.keys()
        finaldim = self.result_q
        m = (
            f'Found {len(self.P)} independent variable products, each ' \
            f'of dimension {finaldim}:\n'
        )
        for i,p in enumerate(self.P):
            # Here we reuse fmt_dimensions as computing the product of variables
            # is the same as computing the derived dimension.
            s = _fmt_dimensions(p, names)
            m = m + f'{i+1:4}: [{s}] = {finaldim}\n'
        return m

class Problem:
    def __init__(self):
        self.__dict__['_variables'] = OrderedDict()

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        pass

    def __setattr__(self, name: str, value: Union[Q, Any]) -> None:
        variables = self.variables
        if isinstance(value, Q):
            _logger.debug(f'Recording variable [{name}] = {value!r}')
            variables[name] = value
        super().__setattr__(name, value)

    @property
    def variables(self) -> OrderedDict:
        return self.__dict__['_variables']

    def solve_for(self, q: Q, **kwargs: dict) -> Result:
        dm = u.dimensional_matrix(self.variables.values())
        info = slv.solver_info(dm, q)
        P = slv.solve(dm, q, info=info, **kwargs)
        return Result(info, q.dimsys, P, copy.deepcopy(self.variables))

def new_problem():
    return Problem()
