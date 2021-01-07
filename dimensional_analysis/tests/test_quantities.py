from ..si import L,M,T,F,P
from numpy.testing import assert_allclose

def test_quantities():
    assert len(L) == 3

    assert str(L) == 'L'
    assert str(M) == 'M'    
    assert str(T) == 'T'    
    assert str(F) == 'LMT(-2)'
    assert str(P) == 'L(-1)MT(-2)'
    assert str(M/M) == '1'

    assert_allclose(L, [1,0,0])
    assert_allclose(M, [0,1,0])
    assert_allclose(T, [0,0,1])
    assert_allclose(F, [1,1,-2])

    assert not F.dimensionless
    assert not L.dimensionless
    assert (F/F).dimensionless
