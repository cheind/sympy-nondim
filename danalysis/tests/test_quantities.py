import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..quantities import DimensionalSystem, Q
from .. import standard_units as si
from .. import utils as u

def test_dimensional_system():
    d = DimensionalSystem(3)
    A,B,C = d.base_quantities()
    assert_allclose(A, [1,0,0])
    assert_allclose(B, [0,1,0])
    assert_allclose(C, [0,0,1])
    assert len(d) == 3
    assert d.base_dims == ['A','B','C']    

    assert_allclose(d.q([0,0,1]), [0,0,1])
    assert_allclose(d.q(), [0,0,0])
    assert_allclose(d.q('B'), [0,1,0])
    assert_allclose(d([0,0,1]), [0,0,1])
    assert_allclose(d(), [0,0,0])
    assert_allclose(d.unity, [0,0,0])
    assert_allclose(d('B'), [0,1,0])
    assert str(d.q([-1,0,2])) == 'A**-1*C**2'
    assert f'{d.q([-1,0,2])!r}' == 'Q(A**-1*C**2)'
    assert d.q().is_dimensionless
    assert not d.q('A').is_dimensionless

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


# def test_quantities_custom_type():
#     Q = Quantity.create_type('Q', 'LMT')
    
#     # Derived quantity classes behave like ordinary quantities
#     # but provide informative dimension names
#     q = Q([1,2,1])    
#     assert len(q) == 3
#     assert str(q) == 'L*M**2*T'
#     assert not q.dimensionless
#     assert q.shape == (3,)  
    
#     qd = (q*[1,0,0]/[0,2,0])**2
#     assert isinstance(qd, Q)
#     assert_allclose(qd, [4,0,2])
#     assert str(qd) == 'L**4*T**2'

#     # Derived quantity classes initialize to unity when no exponents are given
#     u = Q()
#     assert u.dimensionless
#     assert str(u) == '1'

#     # Derived quantity types allow multi-character dimension names
#     R = Quantity.create_type('R', ['DimA','DimB','DimC'])
#     r = (R([1,2,1])*[1,0,0]/[0,2,0])**2
#     assert str(r) == 'DimA**4*DimC**2'


# def test_quantities_unknown_type():
#     q = Quantity([0,0,1])
#     assert len(q) == 3
#     assert str(q) == 'C'
#     assert not q.dimensionless
#     assert q.shape == (3,)    
#     assert str(q/q) == '1'

#     qd = (q*[1,0,0]/[0,2,0])**2
#     assert isinstance(qd, Quantity)
#     assert_allclose(qd, [2,-4, 2])
#     qd = (q*np.asarray([1,0,0])/np.asarray([0,2,0]))**2
#     assert_allclose(qd, [2,-4, 2])
    
#     with pytest.raises(ValueError):
#         Quantity.unity()
#     assert_allclose(Quantity.unity(3), [0,0,0])
#     assert Quantity.unity(3).dimensionless

# def test_quantities_custom_type():
#     Q = Quantity.create_type('Q', 'LMT')
    
#     # Derived quantity classes behave like ordinary quantities
#     # but provide informative dimension names
#     q = Q([1,2,1])    
#     assert len(q) == 3
#     assert str(q) == 'L*M**2*T'
#     assert not q.dimensionless
#     assert q.shape == (3,)  
    
#     qd = (q*[1,0,0]/[0,2,0])**2
#     assert isinstance(qd, Q)
#     assert_allclose(qd, [4,0,2])
#     assert str(qd) == 'L**4*T**2'

#     # Derived quantity classes initialize to unity when no exponents are given
#     u = Q()
#     assert u.dimensionless
#     assert str(u) == '1'

#     # Derived quantity types allow multi-character dimension names
#     R = Quantity.create_type('R', ['DimA','DimB','DimC'])
#     r = (R([1,2,1])*[1,0,0]/[0,2,0])**2
#     assert str(r) == 'DimA**4*DimC**2'

# def test_si():
#     assert len(si.L) == 7
#     assert str(si.L) == 'L'
#     assert str(si.M) == 'M'    
#     assert str(si.T) == 'T'
#     assert str(si.F) == 'L*M*T**-2'
#     assert str(si.pressure) == 'L**-1*M*T**-2'
#     assert str(si.M/si.M) == '1'
#     assert str(si.unity) == '1'

#     assert_allclose(si.L, u.basis_vec(0,7))
#     assert_allclose(si.M, u.basis_vec(1,7))
#     assert_allclose(si.T, u.basis_vec(2,7))
#     assert_allclose(si.F, [1,1,-2,0,0,0,0])

#     assert not si.F.dimensionless
#     assert not si.L.dimensionless
#     assert (si.F/si.F).dimensionless
#     assert (si.F/(si.M*si.L/si.T**2)).dimensionless
