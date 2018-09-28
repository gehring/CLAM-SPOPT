import numpy as np
from itertools import izip
from mathtools import BufferedRowCollector, CompressedMatrix
import copy

import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

class LowRankModel(object):
    def __init__(self, 
                 action_dim,
                 max_rank,
                 phi,
                 lamb = 0.2,
                 trajectories = None,
                 residual_model = False,
                 a_dtype = np.float):
                     
                     
        self.dim = phi.size()
        dim = self.dim
        self.phi = phi

        self.diff_model = residual_model
        self.max_rank = max_rank
        self.lamb = lamb
        self.X_tp1 = BufferedRowCollector(dim)
            
        self.A_t = BufferedRowCollector(action_dim, dtype= a_dtype)
        self.R_t = BufferedRowCollector(1)
            
        self.CompressedX_t = CompressedMatrix(max_rank = max_rank,
                                              size = dim,
                                              deferred_updates = True)
                                              
        if trajectories is not None:
            for S_t, X_t, A_t, R_t, S_tp1, X_tp1, is_terminal in trajectories:
                self.add_sample(X_t, A_t, R_t, X_tp1, is_terminal)

            self.CompressedX_t.update_and_clear_buffer()

    def preprocess_state_size(self):
        return self.dim

    def add_sample(self, X_t, A_t, R_t, X_tp1, is_terminal):
        X_t = self.phi(X_t)
        X_tp1 = self.phi(X_tp1)

        if X_t.ndim == 1:
            X_t = X_t.reshape((1,-1))
        if X_tp1.ndim == 1:
            X_tp1 = X_tp1.reshape((1,-1))
        if A_t.ndim == 1:
            A_t = A_t.reshape((1,-1))

        X_tp1 *= (1.0-is_terminal)[:,None]

        compressed_update = False

        if self.diff_model:
            self.X_tp1.add_rows(X_tp1 - X_t)
        else:
            self.X_tp1.add_rows(X_tp1)
                
        self.A_t.add_rows(A_t)
        self.R_t.add_rows(R_t)
        
        compressed_update |= self.CompressedX_t.add_rows(X_t)
            
        return compressed_update
    @profile
    def get_actions_model(self, w):
        U, S, V = self.CompressedX_t.matrices
        Q,R = np.linalg.qr(w[:,None] * U, mode = 'reduced')
        Up, Sp, VpT = np.linalg.svd(R * S[None, :])
        Ua = Q.dot(Up)
        Sa = Sp
        Va = V.dot(VpT.T)
        return Ua, Sa, Va, w

    @profile
    def generate_embedded_model(self, 
                            action_kernels, 
                            Kab = None, 
                            Da = None, 
                            wa = None,
                            Uas = None,
                            Vas = None,
                            w_ter = None,
                            theta = None,
                            max_rank = None):

        if max_rank is None:
            rank = self.max_rank
        else:
            rank = min(self.max_rank, max_rank)

        rank = min(rank, self.R_t.count)


        d = len(action_kernels)
        if self.CompressedX_t.matrices is None:
            Ma = [None] * d
        else:
            A_kernel = action_kernels(self.A_t.get_matrix(self.CompressedX_t.matrices[0].shape[0]))
            Ma = [ self.get_actions_model(a_k.reshape((-1,))) for a_k in A_kernel]
        

            
        
        if Ma[0] is None:
            # if none, then matrices have not been initialized yet
            # default to predict zero
            Da = np.zeros((d,1))
            wa = np.zeros((d,1))
            Vas = np.zeros((d,self.dim,1))
            Kab = np.zeros((d,d,1,1))
            w_ter = np.zeros_like(wa)

        else:
            if Da is None:
                Da = np.empty((d,rank))
            if wa is None:        
                wa = np.empty((d,rank))
            if Kab is None:
                Kab = np.empty((d,d, rank, rank))
            if Uas is None:
                Uas = np.empty((d,Ma[0][0].shape[0], rank))
            if Vas is None:
                Vas = np.empty((d,self.dim, rank))
            if w_ter is None:
                w_ter = np.empty((d,rank))

            lamb = self.lamb        
            
            R = self.R_t.get_matrix(Ma[0][0].shape[0]).squeeze()

            X_tp1 = self.X_tp1.get_matrix(Ma[0][0].shape[0])

            if theta is not None:
                # w_ter = theta^T V
                # alternative would be to have w_ter = theta^T V \sigma and later multiply by Da
                # w_ter[:,:] = theta.dot(self.CompressedX_t.matrices[2])[None,:]
                theta_tp1 = theta.dot(X_tp1.T)
            else:
                theta_tp1 = np.zeros(X_tp1.shape[0])

            for a in xrange(d):
                Ua, Sa, Va, impor_ratio_a = Ma[a]
                Ua = Ua[:,:rank]
                Sa = Sa[:rank]
                Va = Va[:,:rank]
                
                Vas[a] = Va
                Uas[a] = Ua
                
                Da[a] = Sa/(Sa**2 + lamb)
                wa[a] = (R*impor_ratio_a).squeeze().dot(Ua)
                w_ter[a] = (theta_tp1 * impor_ratio_a).squeeze().dot(Ua)

                for b in xrange(d):
                    Ub, Sb, Vb, impor_ratio_b = Ma[b]
                    Ub = Ub[:,:rank]
                    Xb_tp1 = X_tp1 * impor_ratio_b[:,None]
                    Kab[a,b] = Va.T.dot(Xb_tp1.T.dot(Ub))

                

        return Kab, Da, wa, Uas, Vas, [m[-1] for m in Ma], w_ter

    def update_rewards(self,R,
                            Kab, 
                            Da, 
                            wa,
                            Uas,
                            Vas,
                            imp_a,
                            w_ter):

        d = Da.shape[0]
        for a in xrange(d):
            wa[a] = (R*imp_a[a][:,None]).T.dot(Uas[a])
        return Kab, Da, wa, Uas, Vas, imp_a, w_ter

    def preprocess_states(self, X_t):
        return self.phi(X_t)

    def forceUpdates(self):
        self.CompressedX_t.update_and_clear_buffer()

class FixedLowRankModel(object):

    def __init__(self, ra, Fa, phi):
        Ua, Sa, Va = zip(*Fa)
        self.Fa = [np.concatenate([U[None,:,:] for U in Ua]), np.concatenate([S[None,:] for S in Sa]), np.concatenate([V[None,:,:] for V in Va])]
        self.ra = ra
        self.dim = Fa[0][0].shape[0]
        self.phi = phi

    def generate_embedded_model(self, max_rank, theta = None):
        max_rank = min(self.Fa[1].shape[1], max_rank)
        w_ter = np.zeros(self.Fa[0].shape[1])
        Ua, Sa, Va = self.Fa
        if theta is not None:
            raise NotImplementedError

        return Ua[:,:,:max_rank], Sa[:,:max_rank],Va[:,:,:max_rank], self.ra, w_ter

    def preprocess_states(self, X_t):
        return self.phi(X_t)