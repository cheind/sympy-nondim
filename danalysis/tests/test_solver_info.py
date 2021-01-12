import pytest
from .test_fixtures import *

from ..solver_info import solver_info
from .. import standard_units_min as si

@pytest.mark.usefixtures('vs_example_72')
def test_matrix_solver_info(vs_example_72):
    info = solver_info(vs_example_72, si.unity)
    assert not info.square
    assert info.singular
    assert info.n_d == len(si.unity)
    assert info.n_v == len(vs_example_72)
    assert info.rank == 3
    assert info.delta == 0
    assert info.n_p == len(vs_example_72) - len(si.unity)
    assert info.dimensionless
    assert info.n_s == 3