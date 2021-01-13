import pytest
import numpy as np
from numpy.testing import assert_allclose
from collections import OrderedDict

from .test_fixtures import *
from ..quantities import DimensionalSystem
from ..standard_systems import LMT
from ..problem import new_problem


def test_problem():
    d = DimensionalSystem(3)
    A,B,C = d.base_quantities()

    with new_problem() as p:
        p.v = A             # recorded
        p.x = A**2*B**-1    # recorded
        p.z = 'X'           # not recorded
        assert p.variables == OrderedDict({'v':A, 'x':A**2*B**-1})

@pytest.mark.usefixtures('dm_example_72')
def test_problem_72(dm_example_72):
    L,M,T = LMT.base_quantities()
    with new_problem() as p:
        p.u = L*M**2*T**3,
        p.v = L**2*M**4*T**4
        p.w = (L*M*T)**3
        p.x = L**4*T**2
        p.y = L**5*M**2*T

        r = p.solve_for(LMT.q([3.,5.,7.]))
        assert_allclose(r.result_q, [3., 5., 7.])



