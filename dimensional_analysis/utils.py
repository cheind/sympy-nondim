import numpy as np
from itertools import count

def perm_matrix(perm_indices):
    '''Returns the permutation matrix corresponding to given permutation indices

    Here `perm_indices` defines the permutation order in the following sense: 
    value `j` at index `i` will move row/column `j` of the original matrix to 
    row/column `i`in the permuated matrix P*M/M*P^T.
    
    Params
    ------
    perm_indices: N
        permutation order
    '''    
    N = len(perm_indices)
    pm = np.empty((N,N), dtype=np.int32)
    for i,j in enumerate(perm_indices):
        pm[i] = basis_vec(j, N, dtype=np.int32)
    return pm

def binary_perm_matrix(i, j, N):
    '''Returns permutation matrix that exchanges row/column i and j.'''
    ids = np.arange(N)
    ids[i] = j
    ids[j] = i
    return perm_matrix(ids)

def basis_vec(i, n, dtype=None):
    '''Returns the standard basis vector e_i in R^n.'''
    e = np.zeros(n, dtype=dtype)
    e[i] = 1
    return e

class TrackedMatrixManipulations:
    def __init__(self, m):
        self.R, self.C = m.shape
        self.m = m # Original matrix        
        self.dr = 0 # Number of deleted rows
        self.dc = 0 # Number of deleted cols
        self.rp = np.eye(self.R, dtype=m.dtype) # Sequence of row permutations
        self.cp = np.eye(self.C, dtype=m.dtype) # Sequence of col permutations

    @property
    def matrix(self):
        m = self.rp @ self.m @ self.cp
        return m[:self.R-self.dr, :self.C-self.dc]

    def swap_columns(self, i, j):
        self.cp = self.cp @ binary_perm_matrix(i, j, self.C)
        return self.matrix

    def swap_rows(self, i, j):
        self.rp = binary_perm_matrix(i, j, self.R) @ self.rp
        return self.matrix

    def delete_rows(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        p = self.delete_perm_matrix(ids, rows=True)
        self.rp = p @ self.rp
        self.dr += len(ids)
        return self.matrix
    
    def delete_cols(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        p = self.delete_perm_matrix(ids, rows=False)
        self.cp = self.cp @ p.T # note, transpose
        self.dc += len(ids)
        return self.matrix

    def delete_perm_matrix(self, ids, rows=True):
        '''Returns the permutation matrix that moves deleted rows/columns to the end of the array.'''
        N = self.R if rows else self.C
        d = self.dr if rows else self.dc
        pids = np.empty(N, dtype=np.int32) # each entry holds target row index
        upper = N - d # ignore already deleted ones        
        rcnt = count(upper-1, -1)
        cnt = count(0, 1)
        # We reorder the values i 0..upper in that we assign the value i
        # to index w, where w is chosen from increasing numbers when i is
        # not in the deleted map, otherwise we select w to be the next possible
        # index from the back.
        for i in range(0,upper): 
            w = next(rcnt) if i in ids else next(cnt)
            pids[w] = i
        return perm_matrix(pids)
        
        
