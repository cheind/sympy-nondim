import numpy as np
import logging
from . import utils as u

_logger = logging.getLogger('danalysis')

def _fail(msg, critical=True):    
    if critical:
        _logger.error(msg)
        raise ValueError(msg)
    else:
        _logger.warning(msg)

def assert_log_raise(cond, error):
    if not cond:
        msg = str(error)
        _logger.error(msg)
        raise ValueError(msg)

def assert_zero_q_when_all_zero_rows(dm, q):
    '''Check if all-zero dim-matrix rows correspond to zero entries in q.'''
    assert_log_raise(
        all([q[i]==0. for i in u.zero_rows(dm)]),
        'All-zero rows of dimensional matrix must correspond' \
        'to zero components in q.'
    )

def assert_no_rank_deficit(dmr, qr, rank):
    '''Check for rank deficit in case D > V.'''
    n_d, n_v = dmr.shape
    assert_log_raise(
        not (u.dimensionless(qr) and 
        n_d > n_v and 
        rank > n_v - 1),
        'Rank of dimensional matrix must be <= number of vars - 1' \
        'by Theorem 7-6.'
    )

def assert_square_singular(dmr, qr):
    '''When dmr is square it must be singular when the number of non-zero q components is zero.'''
    n_d, n_v = dmr.shape
    square = n_d == n_v
    assert_log_raise(
        not (square and
        u.dimensionless(qr) and
        np.linalg.det(dmr) != 0),
        'Dimensional matrix must be singular when q is dimensionless '\
        'and number of variables equals number of dimensions. '\
        'No solutions possible, see Theorem 7-5.'
    )

def result_dimension_match(PT, dm, q, strict):
    '''Assert result dimension matches requested dimension.

    They might differ when the number of selectable dimensions is less
    than the number of dimensions. In that case the resulting dimensions
    are fixed by the linear system and thus might differ from what is
    requested in q.
    '''
    rq = PT @ dm.T # NpxNd
    if rq.shape[0] > 0 and not np.allclose(rq[0], q):
        msg = (
            f'Failed to find dimensionality {q}, instead got {rq[0]}.'
        )
        _fail(msg, critical=strict)
        return False
    else:
        return True