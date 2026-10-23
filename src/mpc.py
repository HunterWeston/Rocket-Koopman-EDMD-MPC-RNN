import numpy as np

class LinearMPC:
    def __init__(self, A, B, C, horizon, u_min, u_max, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.N = horizon
        self.u_min = np.asarray(u_min).ravel()
        self.u_max = np.asarray(u_max).ravel()
        self.Q = Q
        self.R = R

        dz = A.shape[0]
        n = C.shape[0]
        du = B.shape[1]
        N = self.N

        Sx_blocks = []
        Su_blocks = []
        Ak = np.eye(dz)
        for k in range(N+1):
            Sx_blocks.append(C @ Ak)
            row = []
            for i in range(N):
                if i < k:
                    row.append(C @ np.linalg.matrix_power(A, k-1-i) @ B)
                else:
                    row.append(np.zeros((n, du)))
            Su_blocks.append(np.hstack(row))
            Ak = A @ Ak

        self.Sx = np.vstack(Sx_blocks)
        self.Su = np.vstack(Su_blocks)

        Qblk = np.kron(np.eye(N+1), Q)
        Rblk = np.kron(np.eye(N), R)
        self.H = self.Su.T @ Qblk @ self.Su + Rblk
        self.Qblk = Qblk
        self.L = float(np.linalg.eigvalsh(self.H).max())
        if self.L < 1e-10:
            self.L = 1.0

        self.u_lb = np.tile(self.u_min, N)
        self.u_ub = np.tile(self.u_max, N)

    def solve(self, z0, ref_traj, u_init=None, iters=300, output_offset=None):
        """
        ref_traj: ((N+1)*n,) stacked reference states
        output_offset: optional stacked physical-state prediction correction
        Returns u_seq: (N*du,) optimal control sequence
        """
        a = self.Sx @ z0
        if output_offset is not None:
            offset = np.asarray(output_offset, dtype=float).ravel()
            if offset.shape != a.shape:
                raise ValueError(f"output_offset must have shape {a.shape}, got {offset.shape}")
            a = a + offset
        g = self.Su.T @ self.Qblk @ (a - ref_traj)
        step = 1.0 / self.L

        if u_init is None:
            u = np.zeros(self.Su.shape[1])
        else:
            u = np.clip(u_init, self.u_lb, self.u_ub)

        for _ in range(iters):
            grad = self.H @ u + g
            u = np.clip(u - step * grad, self.u_lb, self.u_ub)
        return u

