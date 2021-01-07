import numpy as np
from numpy.testing import assert_allclose

from ..si import L,M,T,F,P
from ..analysis import dimensional_matrix, AugmentedMatrix

def test_dimensional_matrix():
    dm = dimensional_matrix(L,M,T)
    assert_allclose(dm, np.eye(3))
    dm = dimensional_matrix(L,F)
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])

def test_augmented_matrix():
    m = np.eye(3); m[1] = 0
    am = AugmentedMatrix(m)
    assert_allclose(am.matrix, m)
    am.remove_rows(am.zero_rows())
    assert_allclose(am.matrix, [[1,0,0],[0,0,1]])
    r,c = am.indices
    assert_allclose(r,[0,2])
    assert_allclose(c,[0,1,2])

    am.swap_columns(1,2)
    am.swap_rows(0,1)
    assert_allclose(am.matrix, [[0,1,0],[1,0,0]])
    r,c = am.indices
    assert_allclose(r,[2,0])
    assert_allclose(c,[0,2,1])