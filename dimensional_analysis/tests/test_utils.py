import numpy as np
from numpy.testing import assert_allclose
from .. import utils as u

def test_remove_rows():
    a = np.arange(10).reshape(5,2)
    aa, ids = u.remove_rows(a, [1,3])
    assert_allclose(aa, [[0,1],[4,5],[8,9]])
    assert_allclose(ids, [0,2,4])
    assert_allclose(a[ids], aa)

def test_permute_columns():
    a = np.arange(10).reshape(2,5)
    perm = [2,1,0,3,4]
    aa = u.permute_columns(a, perm)
    assert_allclose(aa, [[2,1,0,3,4],[7,6,5,8,9]])
    assert_allclose(a[:,perm], aa)