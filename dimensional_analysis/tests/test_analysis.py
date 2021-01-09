import numpy as np
from numpy.testing import assert_allclose

from ..si import L,M,T,F,pressure,one
from .. import analysis as an

def test_dimensional_matrix():
    dm = an.dimensional_matrix(L,M,T)
    assert_allclose(dm, np.eye(3))
    dm = an.dimensional_matrix(L,F)
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])
    dm = an.dimensional_matrix(L*M**2*T**3, L**2*M**4*T**4, (L*M*T)**3,L**4*T**2,L**5*M**2*T)
    assert_allclose(dm, [[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])

def test_matrix_A_B_E():
    m = np.array([[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])
    A, B = an.matrix_A(m), an.matrix_B(m)
    assert_allclose(A, [[3,4,5],[3,0,2],[3,2,1]])
    assert_allclose(B, [[1,2],[2,4],[3,4]])
    E = an.matrix_E(A,B)
    assert_allclose(E*30, [[30,0,0,0,0],[0,30,0,0,0],[-32,-48,-4,6,8],[-6,6,3,-12,9],[18,12,6,6,-12]])
    



    # assert_allclose(an.matrix_E(m), [[1,2],[2,4],[3,4]])




# def test_solve():
#     solve(L,L)
