import pytest
import numpy as np
from numpy.testing import assert_allclose
from .test_fixtures import vs_example_72
from .. import si
from .. import quantities as quant

def test_quantities():
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
