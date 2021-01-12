import pytest
from functools import reduce
import numpy as np
import string
from numpy.testing import assert_allclose

from .test_fixtures import *
from .. import si
from .. import solver as slv
from .. import utils as u



@pytest.mark.usefixtures('vs_example_72')
def test_matrix_A_B_E(vs_example_72):
    info = slv.solver_info(vs_example_72, si.unity)
    A, B = slv._matrix_A(info.dm, info), slv._matrix_B(info.dm, info)
    E = slv._matrix_E(A, B, info)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])

@pytest.mark.usefixtures('vs_example_78')
def test_row_removal_generator(vs_example_78):
    # 3 rows, 2/3 lin. dependent -> one row has to be removed
    info = slv.solver_info(vs_example_78, si.unity)
    order = list(slv._row_removal_generator(info.dm, info))
    assert order == [(0,), (1,), (2,)]
    
    order = list(slv._row_removal_generator(info.dm, info, keep_rows=[0,1]))
    assert order == [(2,), (0,), (1,)]
    

@pytest.mark.usefixtures('vs_example_72')
@pytest.mark.usefixtures('vs_example_78')
def test_ensure_nonsingular_A(vs_example_72, vs_example_78):
    q = si.unity
    info = slv.solver_info(vs_example_72, q)
    del_row, col_order = slv._ensure_nonsingular_A(info.dm, info)
    assert len(del_row) == 0
    assert_allclose(col_order,range(info.n_v))

    q = si.unity
    info = slv.solver_info(vs_example_78, q)
    del_row, col_order = slv._ensure_nonsingular_A(info.dm, info)
    assert len(del_row) == 1
    assert del_row[0] in [1,2]
    assert_allclose(col_order,range(info.n_v))

    info = slv.solver_info([si.M, si.L, si.L], si.unity)
    del_row, col_order = slv._ensure_nonsingular_A(info.dm, info)
    assert len(del_row) == 1
    assert del_row[0] == 2
    assert col_order in [
        (1,0,2),
        (1,2,0),
        (2,0,1),
        (2,1,0)]

def assert_dimensions(P, invars, q):
    s = [reduce(
            lambda prev, z: prev*z[0]**z[1], 
            zip(invars, exponents), 
            si.unity) for exponents in P]
    assert_allclose(s, np.tile(np.asarray(q).reshape(1,-1), (P.shape[0],1)), atol=1e-4)

@pytest.mark.usefixtures('vs_example_72')
def test_solve_72(vs_example_72):
    # No row deletion, no column swap
    q = si.L**3*si.M**5*si.T**7
    P = slv.solve(vs_example_72, q).P
    assert P.shape == (3,5)
    assert_dimensions(P, vs_example_72, q)
    
    # Also try as dict
    var_dict = {n:v for n,v in zip(string.ascii_uppercase, vs_example_72)}
    r = slv.solve(vs_example_72, q)
    assert r.P.shape == (3,5)
    assert_dimensions(r.P, r.info.variables, q)

@pytest.mark.usefixtures('vs_example_78')
def test_solve_78(vs_example_78):
    # Example with a single row deletion
    P = slv.solve(vs_example_78, si.L**2).P
    assert P.shape == (4,5)
    assert_allclose(P, [
        [ 1.,  0.,  0.,  0.,  1.],
        [ 0.,  1.,  0., -1.,  0.],
        [ 0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  0., -1., -1.],
    ])
    assert_dimensions(P, vs_example_78, si.L**2)

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
