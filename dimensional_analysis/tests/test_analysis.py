import pytest
from functools import reduce
import numpy as np
from numpy.testing import assert_allclose

from .. import si
from .. import analysis as an


@pytest.fixture
def vs_example_72():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    return [
        si.L*si.M**2*si.T**3, 
        si.L**2*si.M**4*si.T**4, 
        (si.L*si.M*si.T)**3,
        si.L**4*si.T**2,
        si.L**5*si.M**2*si.T
    ]

@pytest.fixture
def vs_example_78():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    return [
        si.L,
        si.F,
        si.L,
        si.M/(si.L*si.T**2),
        si.L
    ]

def test_dimensional_matrix(vs_example_72):
    dm = an.dimensional_matrix(si.L,si.M,si.T)
    assert_allclose(dm, np.eye(3))
    dm = an.dimensional_matrix(si.L,si.F)
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])    
    dm = an.dimensional_matrix(*vs_example_72)
    assert_allclose(dm, [[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])

def test_matrix_system_meta(vs_example_72):
    dm = an.dimensional_matrix(*vs_example_72)        
    meta = an.DimensionalSystemMeta(dm, si.unity)
    assert not meta.square
    assert meta.singular
    assert meta.n_d == len(si.unity)
    assert meta.n_v == len(vs_example_72)
    assert meta.rank == 3
    assert meta.delta == 0
    assert meta.n_p == len(vs_example_72) - len(si.unity)

def test_matrix_A_B_E(vs_example_72):
    dm = an.dimensional_matrix(*vs_example_72)
    A, B = an.matrix_A(dm), an.matrix_B(dm)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    E = an.matrix_E(A,B)
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])

def test_ensure_nonsingular_A(vs_example_72, vs_example_78):
    dm = an.dimensional_matrix(*vs_example_72)
    q = si.unity
    dm_meta = an.DimensionalSystemMeta(dm, q)
    del_row, col_order = an.ensure_nonsingular_A(dm, dm_meta)
    assert len(del_row) == 0
    assert_allclose(col_order,range(dm_meta.n_v))

    dm = an.dimensional_matrix(*vs_example_78)
    q = si.unity
    dm_meta = an.DimensionalSystemMeta(dm, q)
    del_row, col_order = an.ensure_nonsingular_A(dm, dm_meta)
    assert len(del_row) == 1
    assert del_row[0] in [1,2]
    assert_allclose(col_order,range(dm_meta.n_v))

    dm = an.dimensional_matrix(si.M, si.L, si.L)
    dm_meta = an.DimensionalSystemMeta(dm, si.unity)
    del_row, col_order = an.ensure_nonsingular_A(dm, dm_meta)
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
            zip(invars, p), 
            si.unity) for p in P]
    assert_allclose(s, np.tile(np.asarray(q).reshape(1,-1), (P.shape[0],1)), atol=1e-4)

def test_solve(vs_example_72):
    q = si.L**3*si.M**5*si.T**7
    P = an.solve(vs_example_72, q)
    assert P.shape == (3,5)
    assert_dimensions(P, vs_example_72, q)

    P = an.solve(vs_example_72, si.unity)
    assert P.shape == (2,5)
    assert_dimensions(P, vs_example_72, si.unity)
