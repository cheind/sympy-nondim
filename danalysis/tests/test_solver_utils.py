import pytest
import numpy as np
from numpy.testing import assert_allclose

from .test_fixtures import *
from ..standard_systems import LMT, SI
from .. import meta
from .. import solver_utils as su
from .. import utils as u

@pytest.mark.usefixtures('dm_example_72')
def test_matrix_A_B_E(dm_example_72):
    info = meta.info(dm_example_72, [0.,0.,0.])
    A, B = su.matrix_A(dm_example_72, info), su.matrix_B(dm_example_72, info)
    E = su.matrix_E(A, B, info)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])

@pytest.mark.usefixtures('dm_example_78')
def test_row_removal_generator(dm_example_78):
    # 3 rows, 2/3 lin. dependent -> one row has to be removed
    info = meta.info(dm_example_78, [0.,0.,0.])
    order = list(su.row_removal_generator(dm_example_78, info))
    assert order == [(0,), (1,), (2,)]

@pytest.mark.usefixtures('dm_example_72')
@pytest.mark.usefixtures('dm_example_78')
def test_ensure_nonsingular_A(dm_example_72, dm_example_78):
    info = meta.info(dm_example_72, [0.,0.,0.])
    del_row, col_order = su.ensure_nonsingular_A(dm_example_72, info)
    assert len(del_row) == 0
    assert_allclose(col_order,range(info.n_v))

    info = meta.info(dm_example_78, [0.,0.,0.])
    del_row, col_order = su.ensure_nonsingular_A(dm_example_78, info)
    assert len(del_row) == 1
    assert del_row[0] in [1,2]
    assert_allclose(col_order,range(info.n_v))

    dm = np.zeros((3,3))
    dm[1,0] = 1
    dm[0,1] = 1
    dm[0,2] = 1
    info = meta.info(dm, [0.,0.,0.])
    del_row, col_order = su.ensure_nonsingular_A(dm, info)
    assert len(del_row) == 1
    assert del_row[0] == 2
    assert col_order in [
        [1,0,2],
        [1,2,0],
        [2,0,1],
        [2,1,0]]

    # Test solver-options
    dm = np.zeros((3,3))
    dm[1,0] = 1
    dm[0,1] = 1
    dm[0,2] = 1
    info = meta.info(dm, [0.,0.,0.])
    del_row, col_order = su.ensure_nonsingular_A(dm, info, remove_row_ids=[2], col_perm=[1,2,0])
    assert len(del_row) == 1
    assert del_row[0] == 2
    assert col_order == [1,2,0]

    with pytest.raises(ValueError):
        del_row, col_order = su.ensure_nonsingular_A(dm, info, remove_row_ids=[1], col_perm=[1,2,0])
    with pytest.raises(ValueError):
        del_row, col_order = su.ensure_nonsingular_A(dm, info, col_perm=[0,1,2])
    with pytest.raises(ValueError):
        del_row, col_order = su.ensure_nonsingular_A(dm, info, col_perm=[1,2,1])
