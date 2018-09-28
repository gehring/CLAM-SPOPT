import numpy as np
import ctypes

def int32_to_uint32(i):
    return ctypes.c_uint32(i).value

def get_random_seed(random_stream):
    return int32_to_uint32(random_stream.randint(-2147483648, 2147483648))

def get_soft_max(t, vals, temp = 0.01):
    q = vals/temp
    max_q = q.max()
    rebased_q = q - max_q
    p = np.exp(rebased_q - np.logaddexp.reduce(rebased_q))
    return p

def sample_from_discrete_prob(probs, random_stream):
    bins = np.cumsum(probs)
    return np.digitize(random_stream.random_sample(1), bins)

"""
Return the argmax of an array. If multiple occurence exist, pick a random one
"""
def sampled_argmax(values, random_stream):
    values = values.squeeze()
    indices = np.argwhere(values == np.amax(values))[:,0]
    return random_stream.choice(indices)

def angle_range_check( a, b, x):
    a = np.mod(a, 2*np.pi)
    b = np.mod(b, 2*np.pi)
    theta_bar = np.mod(b-a, 2*np.pi)
    return np.mod(x-a, 2*np.pi)<=theta_bar

class Kernel(object):
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, X, Y):
        raise NotImplementedError()


class DataKernel(object):
    def __init__(self, data, kernel):
        self.data = data
        self.kernel = kernel

    def __call__(self, X):
        return self.kernel(self.data, X)

    def __len__(self):
        return self.data.shape[0]

class RBFKernel(Kernel):
    def __init__(self, scaling):
        self.scaling = scaling

    def __call__(self, X, Y):
        X = X/self.scaling
        Y = Y/self.scaling
        dist = np.sum((X[:,None,:] - Y[None,:,:])**2, axis=2)
        return np.exp(-dist)

class ConstantKernel(Kernel):
    def __init__(self, constant = 1.0):
        self.c = constant

    def __call__(self, X, Y):
        return np.ones((X.shape[0], Y.shape[1]))*self.c


class BufferedRowCollector(object):
    def __init__(self, column_dim, buffer_size = 1000, rows = None, dtype=np.float):
        self.matrix = np.zeros((0, column_dim), dtype=dtype)
        self.buffer = np.zeros((buffer_size, column_dim), dtype=dtype)
        self.count = 0
        if rows is not None:
            self.add_rows(rows)
    def add_rows(self, rows):
        if isinstance(rows, int) or isinstance(rows, float):
            rows = np.array((rows,), dtype = self.matrix.dtype)
        if rows.ndim <= 1:
            if self.buffer.shape[1] > 1:
                rows = rows.reshape((1,-1))
            else:
                rows = rows.reshape((-1,1))
        while(rows.shape[0]>0):
            j = min(rows.shape[0] + self.count, self.buffer.shape[0])
            self.buffer[self.count:j] = rows[:j-self.count,:]
            rows = rows[j-self.count:, :]
            self.count = j
            if j >= self.buffer.shape[0]:
                self.matrix = np.vstack((self.matrix, self.buffer))
                self.count = 0
                
    def get_matrix(self, num_rows = None):
        if (not num_rows is None) and (num_rows > self.matrix.shape[0] + self.count):
            print self.matrix.shape, self.count, self.buffer.shape
            raise Exception('Asked for too many rows, ' \
            + str(num_rows) + ' rows were asked but only ' \
            + str(self.matrix.shape[0] + self.count) + ' rows are stored.')
        if num_rows is None:
            num_rows = self.matrix.shape[0] + self.count
            
        if num_rows > self.matrix.shape[0]:
            self.matrix = np.vstack((self.matrix, self.buffer[:self.count,:]))
            self.count = 0
            
        return self.matrix[:num_rows,:]

class CompressedMatrix(object):
    """ Constructor for a finite rank approximation of a square matrix.
    
        max_rank: maximum allowable rank for the approximation
        
        size: size of the square matrix (i.e., matrix is a size x size)
        
        deferred_updates:   boolean to tell whether updates should always be
                            rank one or if they should be deferred. Currently,
                            when deferred, updates are always of rank 
                            'max_rank'. This could be modified in the future.
    """
    def __init__(self, 
                 max_rank, 
                 size,
                 deferred_updates):
        
        # SVD matrices for the small rank approximation
        self.matrices = None
        
        # flag to know whether updates are deferred
        self.buffered = deferred_updates
        
        # allocate buffer matrices, if need be
        if self.buffered:
            self.ab_buffer = (np.zeros((max_rank, max_rank)),
                              np.zeros((size, max_rank)))
            
        # variable to count the number of buffered updates, not used
        # if we don't defer the updates.
        self.count = 0
        
        # maximum rank of the approximation
        self.max_rank = min(max_rank, size)
        
                
    def add_rows(self, X):
        if not self.buffered:
            raise NotImplementedError()
            
        if X.ndim == 1:
            X = X.reshape((1,-1))
        
        transX = X.T
            
        # save the current update
        A,B = self.ab_buffer
        
        # if the whole update doesn't fit in the buffer, iterate over
        # chunks that do
        model_updated = False
        j = 0
        while j<transX.shape[1]:
            i = self.count + transX.shape[1] - j
            i_clamp = min(i, self.max_rank)
            jp = j + i_clamp - self.count
               
            # place sub-matrix into buffer
            mat_b = transX[:, j:jp]
            B[:, self.count:i_clamp] = mat_b
            if self.matrices is not None:
                n = self.matrices[0].shape[0]
            else:
                n = 0
            A[np.arange(n+self.count,n+i_clamp), np.arange(self.count,i_clamp)] = 1.0
            
            self.count = i_clamp
            j = jp
            
            # if the buffer is full, update svd
            if self.count >= self.max_rank:
                self.update_and_clear_buffer()       
                model_updated = True
                A,B = self.ab_buffer
        return model_updated
            
    """ Update svd with a rank-k outer-product, where k = self.count
    """           
    def update_and_clear_buffer(self):
        if self.count == 0:
            return
            
        A, B = self.ab_buffer
        
        if self.matrices is not None:
            n = self.matrices[0].shape[0]
        else:
            n = 0
                
        self.matrices = self.__update_matrix(A[:n+self.count,:self.count], B[:,:self.count])
        self.matrices = self.__ortho(*self.matrices)
        A = np.zeros((self.matrices[0].shape[0]+self.max_rank, self.matrices[2].shape[0]))
        B[:,:] = 0.0
        self.count = 0
        self.ab_buffer = (A,B)
        return self.matrices
        
    def __ortho(self, U, S, V):
        Qu, Ru = np.linalg.qr(U, mode='reduced')
        Qv, Rv = np.linalg.qr(V, mode='reduced')
        U, S, Vt = np.linalg.svd(Ru.dot(S[:,None] * Rv.T), full_matrices = False)

        return Qu.dot(U), S, Qv.dot(Vt.T)

        
        
    def __update_matrix(self, A, B):
        if self.matrices is not None:
            U,S,V = self.matrices
            U = np.pad(U, ((0,A.shape[0] - U.shape[0]), (0,0)), mode = 'constant')
            Up = U.copy()
            Vp = V.copy()
            Sp = S.copy()
            
            # compute value for the left and right subspace
            Q_a, R_a = np.linalg.qr(A - U.dot(U.T.dot(A)), mode='reduced')
            Q_b, R_b = np.linalg.qr(B - V.dot(V.T.dot(B)), mode='reduced')
    
            Ap = np.vstack((U.T.dot(A), R_a))
            Bp = np.vstack((V.T.dot(B), R_b))
    
            numiters = 5
            success = 1
            for i in range(numiters):
                try:
                    # naive diagonalization of the center matrix (i.e., with SVD)
                    K = np.diag(np.hstack((S, np.zeros(R_a.shape[0])))) + Ap.dot(Bp.T)
                    Up, Sp, Vp = np.linalg.svd(K, full_matrices = False)
                    success = 1
                except:    
                    success = 0
                if success == 1:
                    break
    
            if success == 1:
                # update left and right singular vectors
                U = np.hstack((U, Q_a)).dot(Up)
                V = np.hstack((V, Q_b)).dot(Vp.T)
            
                self.rank = min(self.max_rank, Sp.size)
            else:
                print 'SVD failed to converge, this is not good but we will continue anyway ...\n'
        else:
            # initialize, assuming matrix is all zeroes
            
            # compute value for the left and right subspace
            Q_a, R_a = np.linalg.qr(A, mode='reduced')
            Q_b, R_b = np.linalg.qr(B, mode='reduced')
            
            # initial diagonalization 
            Up, Sp, Vp = np.linalg.svd(R_a.dot(R_b.T), full_matrices = False)

            # construct the singular vectors
            U = Q_a.dot(Up)
            V = Q_b.dot(Vp.T)
            self.initialized = True
            
            # Currently, we don't truncate singular values close to zero and
            # keep the full max_rank approximation. We could potentially remove
            # singular values close to zero.
            self.rank = min(self.max_rank, Sp.size)
            
        S = Sp[:self.rank]
        U = U[:,:self.rank]
        V = V[:,:self.rank]
        return U,S,V

    
    
    """ Build a full, dense matrix of the approximated matrix. This forces
        an update if required.
    """
    def get_updated_full_matrix(self):
        # update SVD if currently using the buffers
        if self.buffered and self.count > 0:
            self.matrices = self.update_and_clear_buffer()
            
        U,S,V = self.matrices
        
        
        return U.dot(np.diag(S).dot(V.T))