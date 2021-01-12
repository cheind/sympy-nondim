import numpy as np
import logging
from . import utils as u

_logger = logging.getLogger('dimensional_analysis')

def assert_log_raise(cond, error):
    if not cond:
        msg = str(error)
        _logger.error(msg)
        raise ValueError(msg)

def assert_zero_q_when_all_zero_rows(dm, q):
    '''Check if all-zero dim-matrix rows correspond to zero entries in q.'''
    assert_log_raise(
        all([q[i]!=0. for i in u.zero_rows(dm)]),
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
        '(Reduced) dimensional matrix must be singular when q is dimensionless'\
        'and number of variables equals number of dimensions.'\
        'See Theorem 7-5.'
    )