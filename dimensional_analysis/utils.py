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

        

class MatrixState:
    def __init__(self, m):
        self.R, self.C = m.shape
        self.m = m
        self.dr = 0 # Number of deleted rows
        self.dc = 0 # Number of deleted cols
        self.rp = np.eye(self.R, dtype=m.dtype) # Sequence of row permutations
        self.cp = np.eye(self.C, dtype=m.dtype) # Sequence of col permutations
        self.history = []

    @property
    def matrix(self):
        '''Returns matrix as represented by the current state'''
        m = self.rp @ self.m @ self.cp
        return m[:self.R-self.dr, :self.C-self.dc]

    @property
    def indices(self):
        '''Returns original row and column indices of the current matrix state.'''
        return np.where(self.rp)[1][:self.R-self.dr], np.where(self.cp.T)[1][:self.C-self.dc]

    def transaction(self):
        return MatrixTransaction(self)

class UndoableMatrixOp:
    def apply(self, state):
        raise NotImplementedError()

    def undo(self, state):
        raise NotImplementedError()

class SwapRowsOp(UndoableMatrixOp):
    def __init__(self, i, j):
        self.ids = (i,j)
        
    def apply(self, state):
        self.p = binary_perm_matrix(self.ids[0], self.ids[1], state.R)
        state.rp = self.p @ state.rp

    def undo(self, state):
        state.rp = self.p.T @ state.rp

class SwapColsOp(UndoableMatrixOp):
    def __init__(self, i, j):
        self.ids = (i,j)
        
    def apply(self, state):
        self.p = binary_perm_matrix(self.ids[0], self.ids[1], state.C)
        state.cp = state.cp @ self.p

    def undo(self, state):
        state.cp = state.cp @ self.p.T


class DeleteOp(UndoableMatrixOp):
    def __init__(self, ids, rows=True):
        if isinstance(ids, int):
            ids = [ids]
        self.ids = ids
        self.rows = rows

    def apply(self, state):
        self.p = DeleteOp.delete_perm_matrix(state, self.ids, rows=self.rows)
        if self.rows:            
            state.rp = self.p @ state.rp
            state.dr += len(self.ids)
        else:
            state.cp = state.cp @ self.p
            state.dc += len(self.ids)

    def undo(self, state):
        if self.rows:            
            state.dr -= len(self.ids)
            state.rp = self.p.T @ state.rp            
        else:
            state.dc -= len(self.ids)
            state.cp = state.cp @ self.p.T        

    @staticmethod
    def delete_perm_matrix(state, ids, rows=True):
        '''Returns the permutation matrix that moves deleted rows/columns to the end of the array.'''
        N = state.R if rows else state.C
        d = state.dr if rows else state.dc
        pids = np.arange(N).astype(dtype=np.int32) # each entry holds target row index
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
        p = perm_matrix(pids)
        return p if rows else p.T

class MatrixTransaction:
    def __init__(self, matrix_state):
        self.matrix_state = matrix_state
        self.committed = None
        self.ops = None

    def __enter__(self):
        self.committed = False
        self.ops = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.committed:
            _undo_all()

    def commit(self):
        self.committed = True

    def swap_rows(self, i, j):
        return self._apply(SwapRowsOp(i, j))

    def swap_cols(self, i, j):
        return self._apply(SwapColsOp(i, j))

    def delete_rows(self, ids):
        return self._apply(DeleteOp(ids, rows=True))

    def delete_cols(self, ids):
        return self._apply(DeleteOp(ids, rows=False))

    def _undo(self):
        op = self.ops.pop()
        op.undo(self)
        return self

    def _undo_all(self):
        while len(self.ops) > 0:
            self.undo()

    def _apply(self, op):
        op.apply(self.matrix_state)
        self.ops.append(op)

def zero_rows(m):
    '''Returns indices of rows containing only zeros.'''
    return np.where(np.all(np.isclose(m, 0), -1))[0]
