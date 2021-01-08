import numpy as np
from numpy.testing import assert_allclose
from .. import utils as u

def test_permutation_matrices():
    assert_allclose(u.perm_matrix([1,0,2]), [[0,1,0],[1,0,0],[0,0,1]])
    m = np.arange(6).astype(np.float32).reshape(2,3)
    assert_allclose(u.binary_perm_matrix(0,1,2) @ m @ u.binary_perm_matrix(0,2,3), [[5,4,3],[2,1,0]])

def test_matrix_ops():
    m = np.arange(9).astype(np.float32).reshape(3,3)

    ms = u.MatrixState(m)
    u.swap_cols(ms,0,2)
    u.swap_rows(ms,0,1)
    u.delete_rows(ms,2)
    assert_allclose(ms.matrix, [[5,4,3],[2,1,0]])
    ms.undo_all()
    assert_allclose(ms.matrix, m)

    m = np.arange(10).astype(np.float32).reshape(5,2)
    ms = u.MatrixState(m)
    u.delete_rows(ms, [2,3,0])
    assert_allclose(ms.matrix, [[2,3],[8,9]])
    assert_allclose(ms.indices[0], [1,4])
    assert_allclose(ms.indices[1], [0,1])
    u.delete_rows(ms, 0)
    assert_allclose(ms.matrix, [[8,9]])
    assert_allclose(ms.indices[0], [4])
    assert_allclose(ms.indices[1], [0,1])
    ms.undo_all()
    assert_allclose(ms.matrix, m)

    m = np.arange(20).astype(np.float32).reshape(4,5)
    ms = u.MatrixState(m)
    u.delete_cols(ms, [1,2,3])
    u.delete_rows(ms, [0,2])
    assert_allclose(ms.matrix, [[5,9],[15,19]])
    assert_allclose(ms.indices[0], [1,3])
    assert_allclose(ms.indices[1], [0,4])
    ms.undo_all()
    assert_allclose(ms.matrix, m)
    