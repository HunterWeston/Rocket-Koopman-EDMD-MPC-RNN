#Hunter Weston

#Linear MPC on lifted space z_{k+1} = A z_k + B u_k, with x_k = C z_k

import numpy as np

class LinearMPC:
    def __init__(self, A, B, C, horizon, u_min, u_max, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.N = horizon
        self.u_min = u_min
        self.u_max = u_max
        Q[0, 0] *= 10
        R *= 0.5 
        self.Q = Q #(n,n)
        self.R = R 

        dz = A.shape[0]
        n = C.shape[0]
        N = self.N
        #build stacked matrices for x = Sx z0 + Su u_seq
        #z_k = A^k z0 + sum_{i=0}^{k-1} A^{k-1-i} B u_i
        Sx_blocks = []
        Su_blocks = []
        Ak = np.eye(dz)
        for k in range(N+1):
            xk_from_z0 = self.C @ Ak
            Sx_blocks.append(xk_from_z0)
            row = []
            for i in range(N):
                if i < k:
                    A_pow = np.linalg.matrix_power(self.A, k-1-i)
                    row.append(self.C @ (A_pow @ self.B))
                else:
                    row.append(np.zeros((n,1)))
            Su_blocks.append(np.hstack(row))
            Ak = self.A @ Ak
        self.Sx = np.vstack(Sx_blocks) #((N+1)*n, dz)
        self.Su = np.vstack(Su_blocks) #((N+1)*n, N)

        Qblk = np.kron(np.eye(N+1), self.Q)
        Rblk = self.R * np.eye(N)
        self.Qblk = Qblk
        self.Rblk = Rblk

    def solve(self, z0, ref_traj, u_init=None, iters=500, step=1e-4):
        """
        Solve min_u 0.5*(Sx z0 + Su u - r)^T Qblk (Sx z0 + Su u - r) + 0.5*u^T Rblk u
        s.t. u_min <= u_i <= u_max
        Uses projected gradient iterations
        """
        nblk = self.Qblk.shape[0]
        a = self.Sx @ z0
        H = self.Su.T @ self.Qblk @ self.Su + self.Rblk
        g = self.Su.T @ self.Qblk @ (a - ref_traj)

        if u_init is None:
            u = np.zeros((self.Su.shape[1],))
        else:
            u = u_init.copy()

        #estimate Lipschitz const
        L = np.linalg.norm(H, 2)
        if L > 0:
            step = 1.0 / L
        for _ in range(iters):
            grad = H @ u + g
            u = u - step * grad
            u = np.clip(u, self.u_min, self.u_max)
        return u

def build_reference(cfg, N):
    #const ref to (h_ref, v_ref, m_current)
    ref = np.zeros(((N+1)*3,))
    for k in range(N+1):
        ref[3*k + 0] = cfg.ctrl.ref_altitude
        ref[3*k + 1] = cfg.ctrl.ref_velocity
        ref[3*k + 2] = 0.0
    return ref
