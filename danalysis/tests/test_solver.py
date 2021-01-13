import pytest
#from functools import reduce
import numpy as np
from numpy.testing import assert_allclose

from .test_fixtures import *
from ..standard_systems import LMT, SI
from .. import solver as slv
from .. import utils as u

@pytest.mark.usefixtures('dm_example_72')
def test_matrix_A_B_E(dm_example_72):
    info = slv.solver_info(dm_example_72, [0.,0.,0.])
    A, B = slv._matrix_A(dm_example_72, info), slv._matrix_B(dm_example_72, info)
    E = slv._matrix_E(A, B, info)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])

@pytest.mark.usefixtures('dm_example_78')
def test_row_removal_generator(dm_example_78):
    # 3 rows, 2/3 lin. dependent -> one row has to be removed
    info = slv.solver_info(dm_example_78, [0.,0.,0.])
    order = list(slv._row_removal_generator(dm_example_78, info))
    assert order == [(0,), (1,), (2,)]
    
    order = list(slv._row_removal_generator(dm_example_78, info, keep_rows=[0,1]))
    assert order == [(2,), (0,), (1,)]
    

@pytest.mark.usefixtures('dm_example_72')
@pytest.mark.usefixtures('dm_example_78')
def test_ensure_nonsingular_A(dm_example_72, dm_example_78):
    info = slv.solver_info(dm_example_72, [0.,0.,0.])
    del_row, col_order = slv._ensure_nonsingular_A(dm_example_72, info)
    assert len(del_row) == 0
    assert_allclose(col_order,range(info.n_v))

    info = slv.solver_info(dm_example_78, [0.,0.,0.])
    del_row, col_order = slv._ensure_nonsingular_A(dm_example_78, info)
    assert len(del_row) == 1
    assert del_row[0] in [1,2]
    assert_allclose(col_order,range(info.n_v))

    dm = np.zeros((3,3))
    dm[1,0] = 1
    dm[0,1] = 1
    dm[0,2] = 1
    info = slv.solver_info(dm, [0.,0.,0.])
    del_row, col_order = slv._ensure_nonsingular_A(dm, info)
    assert len(del_row) == 1
    assert del_row[0] == 2
    assert col_order in [
        (1,0,2),
        (1,2,0),
        (2,0,1),
        (2,1,0)]

def test_solve_e_has_zero_rows():
    # Number of solutions is 1 which makes e zero rows (no variables to choose freely).
    dm = np.array([
        [0.,1,0],   # M
        [1,1,-2],   # F
        [0,0,1]     # T
    ]).T # DxV
    P = slv.solve(dm, [1.,0, 0]) # PxV
    assert P.shape == (1,3)
    assert_allclose(P @ dm.T, [[1.,0, 0]]) # PxD

@pytest.mark.usefixtures('dm_example_72')
def test_solve_72(dm_example_72):
    # No row deletion, no column swap
    P = slv.solve(dm_example_72, [3., 5., 7.])
    assert P.shape == (3,5)
    assert_allclose(P @ dm_example_72.T, np.tile([[3.,5.,7.]], (3,1))) # PxD
        
@pytest.mark.usefixtures('dm_example_78')
def test_solve_78(dm_example_78):
    # Single row deletion
    P = slv.solve(dm_example_78, [2., 0, 0.])
    assert P.shape == (4,5)
    assert_allclose(P, [
        [ 1.,  0.,  0.,  0.,  1.],
        [ 0.,  1.,  0., -1.,  0.],
        [ 0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  0., -1., -1.],
    ])
    assert_allclose(P @ dm_example_78.T, np.tile([[2.,0.,0.]], (4,1))) # PxD
    
    # info = slv.solver_info(vs_example_78, si.M**2)
    # assert info.n_s == 2
    # P = slv.solve(vs_example_78, si.M**2, keep_rows=[0,1])
    # from ..io import fmt_solution
    # print(fmt_solution(P, vs_example_78))

    # P = slv.solve(dm, si.T**2, keep_rows=[0,2])
    # from ..io import fmt_solution
    # print(fmt_solution(P, vs_example_78))
    # print(P)
    # # assert P.shape == (4,5)
    # # assert_allclose(P, [
    # #     [ 1.,  0.,  0.,  0.,  1.],
    # #     [ 0.,  1.,  0., -1.,  0.],
    # #     [ 0.,  0.,  1.,  0.,  1.],
    # #     [ 1.,  1.,  0., -1., -1.],
    # # ])
    # assert_dimensions(P, vs_example_78, si.M**2)


    # from ..io import fmt_solution
    # print(fmt_solution(P, vs_example_78))



    # what if we took q==si.M**2 which corresponds to the row being deleted?
    # -> q becomes dimensionless
