import numpy as np
from numpy.testing import assert_allclose

from .test_fixtures import *
from .. import utils as u
from ..standard_systems import LMT

def test_remove_rows():
    a = np.arange(10).reshape(5,2)
    aa, ids = u.remove_rows(a, [1,3])
    assert_allclose(aa, [[0,1],[4,5],[8,9]])
    assert_allclose(ids, [0,2,4])
    assert_allclose(a[ids], aa)

def test_permute_columns():
    a = np.arange(10).reshape(2,5)
    perm = [2,1,0,3,4]
    aa, inv_perm = u.permute_columns(a, perm)
    assert_allclose(aa, [[2,1,0,3,4],[7,6,5,8,9]])
    assert_allclose(a[:,perm], aa)
    assert_allclose(aa[:,inv_perm], a)

def test_dimensional_matrix():
    L,M,T = LMT.base_quantities()
    dm = u.dimensional_matrix([L,M,T])
    assert_allclose(dm, np.eye(3))
    dm = u.dimensional_matrix([L,LMT.F])
    assert_allclose(dm, [[1, 1],[0, 1], [0, -2]])    

    vs_72 = [
        L*M**2*T**3, 
        L**2*M**4*T**4, 
        (L*M*T)**3,
        L**4*T**2,
        L**5*M**2*T
    ]
    dm = u.dimensional_matrix(vs_72)
    assert_allclose(dm, [[1,2,3,4,5],[2,4,3,0,2],[3,4,3,2,1]])


def test_variable_product():        
    L,M,T = LMT.base_quantities()
    v = u.variable_product([L,M,T], [2,1,2])
    assert_allclose(v, [2,1,2])
    v = u.variable_product([LMT.F,LMT.F], [1,-1])
    assert_allclose(v, LMT.Unity)
