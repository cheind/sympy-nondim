import numpy as np
from itertools import count

def basis_vec(i, n, dtype=None):
    '''Returns the standard basis vector e_i in R^n.'''
    e = np.zeros(n, dtype=dtype)
    e[i] = 1
    return e

def zero_rows(m):
    '''Returns indices of rows containing only zeros.'''
    return np.where(np.all(np.isclose(m, 0), -1))[0]

def remove_rows(a, ids):
    '''Remove the rows in `ids` from array `a`.

    Params
    ------
    a : array
        Array
    ids: array-like
        Indices of rows to remove
    
    Returns
    -------
    m : array
        input array with rows in ids removed
    ids: array-like
        original indices of remaining rows
    '''
    mask = np.ones(a.shape[0], dtype=bool)    
    mask[list(ids)]=0    
    return a[mask], np.arange(a.shape[0])[mask]

def permute_columns(a, perm):
    '''Permute columns of array.
    
    Params
    ------
    a : MxN array
        Array
    perm : N array-like
        New order of indices.

    Returns
    -------
    m : MxN array
        Permuted array        
    inv_perm: N array-like
        Inverse permutation order
    '''
    a = np.atleast_2d(a)
    return a[:, perm], np.argsort(perm)


def dimensionless(q):
    '''Returns if a quantity is dimensionless.'''
    return np.allclose(q, 0.)

def dimensional_matrix(dvars):
    '''Returns the dimensional matrix formed by the given variables.'''
    cols = [np.asarray(v) for v in dvars]
    return np.stack(cols, -1)