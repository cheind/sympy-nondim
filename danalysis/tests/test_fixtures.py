import pytest

from .. import standard_units_min as si
from ..quantities import Quantity

@pytest.fixture
def vs_example_72():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    return [
        si.L*si.M**2*si.T**3, 
        si.L**2*si.M**4*si.T**4, 
        (si.L*si.M*si.T)**3,
        si.L**4*si.T**2,
        si.L**5*si.M**2*si.T
    ]

@pytest.fixture
def vs_example_78():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    return [
        si.L,
        si.F,
        si.L,
        si.M/(si.L*si.T**2),
        si.L
    ]