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
        #build regressor phi = [z u]
        Phi = np.hstack([Z, U])  #shape:(n, dz+du)
        #solve theta^t = (phi^t phi + lam i)^-1 phi^t zp
        lam = self.ridge
        G = Phi.T @ Phi + lam*np.eye(Phi.shape[1])
        RHS = Phi.T @ Zp
        Theta_T = np.linalg.solve(G, RHS)
        Theta = Theta_T.T  #shape:(dz, dz+du)
        self.A = Theta[:, :dz]
        self.B = Theta[:, dz:]
        return self.A, self.B

