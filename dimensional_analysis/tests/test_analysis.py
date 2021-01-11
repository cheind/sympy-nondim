import pytest
from functools import reduce
import numpy as np
from numpy.testing import assert_allclose

from .test_fixtures import *
from .. import si
from .. import analysis as an
from .. import utils as u

@pytest.mark.usefixtures('vs_example_72')
def test_matrix_solver_info(vs_example_72):
    dm = u.dimensional_matrix(vs_example_72)        
    info = an.solver_info(dm, si.unity)
    assert not info.square
    assert info.singular
    assert info.n_d == len(si.unity)
    assert info.n_v == len(vs_example_72)
    assert info.rank == 3
    assert info.delta == 0
    assert info.n_p == len(vs_example_72) - len(si.unity)
    assert info.dimensionless
    assert info.n_s == 3

@pytest.mark.usefixtures('vs_example_72')
def test_matrix_A_B_E(vs_example_72):
    dm = u.dimensional_matrix(vs_example_72)
    A, B = an.matrix_A(dm), an.matrix_B(dm)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    
    E = an.matrix_E(A, B, an.solver_info(dm, si.unity))
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])

@pytest.mark.usefixtures('vs_example_78')
def test_row_removal_generator(vs_example_78):
    # 3 rows, 2/3 lin. dependent -> one row has to be removed
    dm = u.dimensional_matrix(vs_example_78)
    info = an.solver_info(dm, si.unity)
    order = list(an.row_removal_generator(dm, info))
    assert order == [(0,), (1,), (2,)]
    
    order = list(an.row_removal_generator(dm, info, keep_rows=[0,1]))
    assert order == [(2,), (0,), (1,)]
    

@pytest.mark.usefixtures('vs_example_72')
@pytest.mark.usefixtures('vs_example_78')
def test_ensure_nonsingular_A(vs_example_72, vs_example_78):
    dm = u.dimensional_matrix(vs_example_72)
    q = si.unity
    info = an.solver_info(dm, q)
    del_row, col_order = an.ensure_nonsingular_A(dm, info)
    assert len(del_row) == 0
    assert_allclose(col_order,range(info.n_v))

    dm = u.dimensional_matrix(vs_example_78)
    q = si.unity
    info = an.solver_info(dm, q)
    del_row, col_order = an.ensure_nonsingular_A(dm, info)
    assert len(del_row) == 1
    assert del_row[0] in [1,2]
    assert_allclose(col_order,range(info.n_v))

    dm = u.dimensional_matrix([si.M, si.L, si.L])
    info = an.solver_info(dm, si.unity)
    del_row, col_order = an.ensure_nonsingular_A(dm, info)
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
    P = an.solve(u.dimensional_matrix(vs_example_72), q)
    assert P.shape == (3,5)
    assert_dimensions(P, vs_example_72, q)

    P = an.solve(u.dimensional_matrix(vs_example_72), si.unity)
    assert P.shape == (2,5)
    assert_dimensions(P, vs_example_72, si.unity)

@pytest.mark.usefixtures('vs_example_78')
def test_solve_78(vs_example_78):
    # Example with a single row deletion
    print(u.dimensional_matrix(vs_example_78))
    P = an.solve(u.dimensional_matrix(vs_example_78), si.L**2)
    assert P.shape == (4,5)
    assert_allclose(P, [
        [ 1.,  0.,  0.,  0.,  1.],
        [ 0.,  1.,  0., -1.,  0.],
        [ 0.,  0.,  1.,  0.,  1.],
        [ 1.,  1.,  0., -1., -1.],
    ])
    assert_dimensions(P, vs_example_78, si.L**2)

    dm = u.dimensional_matrix(vs_example_78)
    info = an.solver_info(dm, si.M**2)
    assert info.n_s == 2
    P = an.solve(dm, si.M**2, keep_rows=[0,2])
    from ..io import fmt_solution
    print(fmt_solution(P, vs_example_78))

    print(P)
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
