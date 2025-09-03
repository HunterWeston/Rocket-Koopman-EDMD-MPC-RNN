#Hunter Weston

#Polynomial lifting (up to degree d) for EDMD
#Outputs learned C matrix for projecting lifted state to physical state

import numpy as np

def monomials(x, degree=2, include_bias=True):
    x = np.asarray(x).ravel()
    n = x.shape[0]
    feats = []
    if include_bias:
        feats.append(1.0)
    #degree 1
    feats.extend(list(x))
    if degree >= 2:
        for i in range(n):
            for j in range(i, n):
                feats.append(x[i]*x[j])
    if degree >= 3:
        #cubic terms
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    feats.append(x[i]*x[j]*x[k])
    return np.array(feats, dtype=float)

class LiftingMap:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        #dimension inferred on first call

    def lift(self, X):
        """
        X: shape (N, n)
        returns Z: shape (N, d_z)
        """
        X = np.atleast_2d(X)
        Z_list = [monomials(x, self.degree, self.include_bias) for x in X]
        Z = np.vstack(Z_list)
        return Z

def fit_C(Z, X):
    """
    Fits linear map C such that X ≈ Z @ C^T
    Z: (N, d_z), X: (N, n)
    returns C: (n, d_z) s.t. X ≈ (C @ z)_T
    """
    lam = 1e-8
    ZTZ = Z.T @ Z + lam*np.eye(Z.shape[1])
    C_T = np.linalg.solve(ZTZ, Z.T @ X)
    C = C_T.T
    return C
