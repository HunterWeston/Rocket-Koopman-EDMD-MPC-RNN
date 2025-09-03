#Hunter Weston

#EDMD w control observable: z_{k+1} ≈ A z_k + B u_k
#Ridge regularized least squares fit
#Online updates with sliding window

import numpy as np

class EDMDc:
    def __init__(self, ridge_lambda=1e-4, dz=None, du=1):
        self.ridge = ridge_lambda
        self.A = None
        self.B = None
        self.dz = dz
        self.du = du

    def fit(self, Z, U, Zp):
        """
        Fit A, B from data: Z (N, dz), U (N, du), Zp (N, dz)
        Solves Zp ≈ Z A^T + U B^T with ridge regularization
        """
        N, dz = Z.shape
        self.dz = dz
        self.du = U.shape[1]
        #build regressor Phi = [Z U]
        Phi = np.hstack([Z, U])  # (N, dz+du)
        #solve Theta^T = (Phi^T Phi + lam I)^{-1} Phi^T Zp
        lam = self.ridge
        G = Phi.T @ Phi + lam*np.eye(Phi.shape[1])
        RHS = Phi.T @ Zp
        Theta_T = np.linalg.solve(G, RHS)
        Theta = Theta_T.T  #(dz, dz+du)
        self.A = Theta[:, :dz]
        self.B = Theta[:, dz:]
        return self.A, self.B

    def predict(self, z, u):
        return self.A @ z + self.B @ u

class OnlineBuffer:
    def __init__(self, dz, du, maxlen=1000):
        self.maxlen = maxlen
        self.Z = np.zeros((0, dz))
        self.U = np.zeros((0, du))
        self.Zp = np.zeros((0, dz))

    def append(self, z, u, zp):
        self.Z = np.vstack([self.Z, z.reshape(1,-1)])
        self.U = np.vstack([self.U, u.reshape(1,-1)])
        self.Zp = np.vstack([self.Zp, zp.reshape(1,-1)])
        if self.Z.shape[0] > self.maxlen:
            excess = self.Z.shape[0] - self.maxlen
            self.Z = self.Z[excess:]
            self.U = self.U[excess:]
            self.Zp = self.Zp[excess:]
