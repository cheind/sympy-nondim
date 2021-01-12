import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..quantities import Quantity
from .. import si
from .. import utils as u


def test_quantities_unknown_type():
    q = Quantity([0,0,1])
    assert len(q) == 3
    assert str(q) == 'C'
    assert not q.dimensionless
    assert q.shape == (3,)    
    assert str(q/q) == '1'

    qd = (q*[1,0,0]/[0,2,0])**2
    assert isinstance(qd, Quantity)
    assert_allclose(qd, [2,-4, 2])
    qd = (q*np.asarray([1,0,0])/np.asarray([0,2,0]))**2
    assert_allclose(qd, [2,-4, 2])
    
    with pytest.raises(ValueError):
        Quantity.unity()
    assert_allclose(Quantity.unity(3), [0,0,0])
    assert Quantity.unity(3).dimensionless

def test_quantities_custom_type():
    Q = Quantity.create_type('Q', 'LMT')
    
    # Derived quantity classes behave like ordinary quantities
    # but provide informative dimension names
    q = Q([1,2,1])    
    assert len(q) == 3
    assert str(q) == 'L*M**2*T'
    assert not q.dimensionless
    assert q.shape == (3,)  
    
    qd = (q*[1,0,0]/[0,2,0])**2
    assert isinstance(qd, Q)
    assert_allclose(qd, [4,0,2])
    assert str(qd) == 'L**4*T**2'

    # Derived quantity classes initialize to unity when no exponents are given
    u = Q()
    assert u.dimensionless
    assert str(u) == '1'

    # Derived quantity types allow multi-character dimension names
    R = Quantity.create_type('R', ['DimA','DimB','DimC'])
    r = (R([1,2,1])*[1,0,0]/[0,2,0])**2
    assert str(r) == 'DimA**4*DimC**2'

def test_si():
    assert len(si.L) == 3

    assert str(si.L) == 'L'
    assert str(si.M) == 'M'    
    assert str(si.T) == 'T'    
    assert str(si.F) == 'L*M*T**-2'
    assert str(si.pressure) == 'L**-1*M*T**-2'
    assert str(si.M/si.M) == '1'
    assert str(si.unity) == '1'

    assert_allclose(si.L, [1,0,0])
    assert_allclose(si.M, [0,1,0])
    assert_allclose(si.T, [0,0,1])
    assert_allclose(si.F, [1,1,-2])

    assert not si.F.dimensionless
    assert not si.L.dimensionless
    assert (si.F/si.F).dimensionless
    assert (si.F/(si.M*si.L/si.T**2)).dimensionless
