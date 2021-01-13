import pytest

from .test_fixtures import *
from .. import solver as slv

@pytest.mark.usefixtures('dm_example_72')
def test_matrix_solver_info(dm_example_72):
    info = slv.solver_info(dm_example_72, [0.,0.,0.])
    assert not info.square
    assert info.singular
    assert info.n_d == 3
    assert info.n_v == dm_example_72.shape[1]
    assert info.rank == 3
    assert info.delta == 0
    assert info.n_p == 5 - 3
    assert info.dimensionless
    assert info.n_s == 3