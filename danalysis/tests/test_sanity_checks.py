import pytest
import numpy as np
from .. import sanity_checks as checks

def test_assert_zero_q_when_all_zero_rows():
    with pytest.raises(ValueError):
        dm = np.eye(3)
        dm[2] = 0
        q = np.zeros(3)
        q[2] = 0.1
        checks.assert_zero_q_when_all_zero_rows(dm, q)

    # This should be ok
    dm = np.eye(3)
    dm[2] = 0
    q = np.zeros(3)
    checks.assert_zero_q_when_all_zero_rows(dm, q)
