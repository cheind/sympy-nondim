import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..quantities import DimensionalSystem, Q
from ..standard_systems import SI, LMT
from .. import utils as u

def test_dimensional_system():
    d = DimensionalSystem(3)
    A,B,C = d.base_quantities()
    assert_allclose(A, [1,0,0])
    assert_allclose(B, [0,1,0])
    assert_allclose(C, [0,0,1])
    # Base quantities are also exposed in d itself.
    assert_allclose(d.A, [1,0,0])
    assert_allclose(d.B, [0,1,0])
    assert_allclose(d.C, [0,0,1])
    assert len(d) == 3
    assert d.base_dims == ['A','B','C']    

    assert_allclose(d.q([0,0,1]), [0,0,1])
    assert_allclose(d.q(), [0,0,0])
    assert_allclose(d.q('B'), [0,1,0])
    assert_allclose(d([0,0,1]), [0,0,1])
    assert_allclose(d(), [0,0,0])
    assert_allclose(d.Unity, [0,0,0])
    assert_allclose(d('B'), [0,1,0])
    assert str(d.q([-1,0,2])) == 'A**-1*C**2'
    assert f'{d.q([-1,0,2])!r}' == 'Q(A**-1*C**2)'
    assert d.q().is_dimensionless
    assert not d.q('A').is_dimensionless

    # Decompose
    dm = np.random.randn(3,4)
    qs = d.qs_from_dm(dm)
    assert len(qs) == 4
    assert_allclose(qs[0], dm[:,0])
    assert_allclose(qs[1], dm[:,1])
    assert_allclose(qs[2], dm[:,2])
    assert_allclose(qs[3], dm[:,3])

    # Math
    qd = (d.q('C')*[1,0,0]/[0,2,0])**2
    assert isinstance(qd, Q)
    assert_allclose(qd, [2,-4,2])
    assert (qd/qd).is_dimensionless

    # Dimensional system with named dims
    lmt = DimensionalSystem('LMT')
    L,M,T = lmt.base_quantities()
    assert_allclose(L, [1,0,0])
    assert_allclose(M, [0,1,0])
    assert_allclose(T, [0,0,1])
    assert lmt.base_dims == ['L','M','T']

    # Dimensional system with named dims
    ab = DimensionalSystem(['DimA', 'DimB'])
    A,B = ab.base_quantities()
    assert_allclose(A, [1,0])
    assert_allclose(B, [0,1])
    assert ab.base_dims == ['DimA','DimB']

    assert not lmt == d
    assert ab == DimensionalSystem(['DimA', 'DimB'])

def test_si():
    assert len(SI) == 7
    dm = u.dimensional_matrix(SI.base_quantities())    
    assert_allclose(dm, np.eye(7))

    assert len(LMT) == 3
    dm = u.dimensional_matrix(LMT.base_quantities())    
    assert_allclose(dm, np.eye(3))