import numpy as np
from numpy.testing import assert_allclose
from .. import utils as u

def test_permutation_matrices():
    assert_allclose(u.perm_matrix([1,0,2]), [[0,1,0],[1,0,0],[0,0,1]])
    m = np.arange(6).astype(np.float32).reshape(2,3)
    assert_allclose(u.binary_perm_matrix(0,1,2) @ m @ u.binary_perm_matrix(0,2,3), [[5,4,3],[2,1,0]])

def test_tracked_matrix():
    m = np.arange(9).astype(np.float32).reshape(3,3)
    tm = u.TrackedMatrixManipulations(m)
    tm.swap_columns(0,2)
    tm.swap_rows(0,1)
    tm.delete_rows(2)
    assert_allclose(tm.matrix, [[5,4,3],[2,1,0]])

    m = np.arange(10).astype(np.float32).reshape(5,2)
    tm = u.TrackedMatrixManipulations(m)
    tm.delete_rows([2,3,0])
    assert_allclose(tm.matrix, [[2,3],[8,9]])
    tm.delete_rows(0)
    assert_allclose(tm.matrix, [[8,9]])

    m = np.arange(20).astype(np.float32).reshape(4,5)
    tm = u.TrackedMatrixManipulations(m)
    tm.delete_cols([1,2,3])
    tm.delete_rows([0,2])
    assert_allclose(tm.matrix, [[5,9],[15,19]])