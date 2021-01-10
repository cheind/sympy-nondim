import numpy as np
from numpy.testing import assert_allclose

from .. import si
from .. import analysis as an

def test_dimensional_matrix():
    dm = an.dimensional_matrix(si.L,si.M,si.T)
    assert_allclose(dm, np.eye(3))
    dm = an.dimensional_matrix(si.L,si.F)
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    dm = an.dimensional_matrix(si.L*si.M**2*si.T**3, si.L**2*si.M**4*si.T**4, (si.L*si.M*si.T)**3,si.L**4*si.T**2,si.L**5*si.M**2*si.T)
    assert_allclose(dm, [[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])

def test_matrix_A_B_E():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    m = np.array([[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])
    A, B = an.matrix_A(m), an.matrix_B(m)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    E = an.matrix_E(A,B)
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])





# def test_solve():
#     solve(L,L)
