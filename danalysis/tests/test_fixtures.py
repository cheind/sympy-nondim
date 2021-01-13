import pytest
import numpy as np


@pytest.fixture
def dm_example_72():
    # Example 7-2 pp. 137 of Applied Dimensional Analysis and Modeling 
    return np.array([
        [1.,2,3],
        [2,4,4],
        [3,3,3],
        [4,0,2],
        [5,2,1]
    ]).T

    # return [
    #     si.L*si.M**2*si.T**3, 
    #     si.L**2*si.M**4*si.T**4, 
    #     (si.L*si.M*si.T)**3,
    #     si.L**4*si.T**2,
    #     si.L**5*si.M**2*si.T
    # ]

@pytest.fixture
def dm_example_78():
    # Example 7-2 pp. 146 of Applied Dimensional Analysis and Modeling 
    return np.array([
        [1., 0, 0],
        [1, 1, -2],
        [1, 0, 0],
        [-1, 1, -2],
        [1, 0, 0]
    ]).T
    # return [
    #     si.L,
    #     si.F,
    #     si.L,
    #     si.M/(si.L*si.T**2),
    #     si.L
    # ]