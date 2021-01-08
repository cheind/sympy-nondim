import numpy as np
from numpy.testing import assert_allclose

from ..si import L,M,T,F,P
from ..analysis import dimensional_matrix

def test_dimensional_matrix():
    dm = dimensional_matrix(L,M,T)
    assert_allclose(dm, np.eye(3))
    dm = dimensional_matrix(L,F)
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])