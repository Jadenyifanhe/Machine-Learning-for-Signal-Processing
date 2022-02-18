
import numpy as np
import scipy
from scipy import linalg


class FOBIICA():

    def __init__(self, n_components=None):
        self.mixing_ = None
        self.n_components = n_components

    def _fit(self, X, compute_sources=False):

        X_mean = np.mean(X, axis=1, keepdims=True)
        X = X - X_mean
        X_ = X.T[:, :, np.newaxis]
        X_T_ = np.transpose(X_, (0, 2, 1))
        X_XT = matmul(X_, X_T_)

        eigen_values, eigen_vectors = orth_decomposition((X_XT + X_XT.T)/2)
        sigma = eigen_values
        U_P = eigen_vectors

        X_hat = np.diag(1/np.sqrt(sigma)) @ U_P.T @ X
        X_hat = X_hat.real if X_hat.dtype == 'complex' else X_hat
        X_hat_ = X_hat.T[:, :, np.newaxis]
        X_hat_T_ = np.transpose(X_hat_, (0, 2, 1))

        D_X_hat = X_hat_T_[0] @ X_hat_[0] * X_hat_[0] @ X_hat_T_[0]
        for i in range(1, X_hat_.shape[0]):
            D_X_hat = (D_X_hat * i + X_hat_T_[i] @ X_hat_[i] * X_hat_[i] @ X_hat_T_[i]) / (i + 1)

        eigenvalues_D, eigenvectors_D = orth_decomposition((D_X_hat + D_X_hat.T)/2)

        W = eigenvectors_D.T.real if eigenvectors_D.dtype == 'complex' else eigenvectors_D.T
        S = W @ X_hat
        s_pinv = scipy.linalg.pinv(S)
        mixing_ = s_pinv @ X
        self.mixing_ = mixing_.T
        return S

    def fit_transform(self, X, y=None):

        return self._fit(X, compute_sources=True)


def checking(matrix, eigenvalue, eigenvector):
    # just for checking
    matrix_recons = eigenvector @ np.diag(eigenvalue) @ eigenvector.T
    delta_matrix = matrix - matrix_recons
    delta_proj = matrix @ eigenvector - eigenvalue * eigenvector
    return delta_matrix, delta_proj


def matmul(A, B):
    # A, B: (n, rows, cols)
    try:
        C = A @ B  # may lead to OOM!
        C = np.mean(C, axis=0)
    except:
        n = len(A)
        C = A[0] @ B[0]
        for i in range(1, n):
            C = (C * i + A[i] @ B[i]) / (i + 1)
    return C


# def eigen_decomposition(X, sort=True, check=True):
#     eigen_values, eigen_vectors = scipy.linalg.eig(X)
#     # eigen_values = eigen_values.real
#     # eigen_vectors = eigen_vectors.real
#
#     if sort:
#         idx = eigen_values.argsort()[::-1]
#         eigen_values = eigen_values[idx]
#         eigen_vectors = eigen_vectors[:, idx]
#
#     if check:
#         delta_X, delta_proj_X = checking(X, eigen_values, eigen_vectors)
#     return eigen_values, eigen_vectors


def orth_decomposition(X, sort=True, check=True):
    assert ((X-X.T)!=0.0).astype(float).sum() == 0
    eigen_values, eigen_vectors = scipy.linalg.eig(X)

    if sort:
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

    if check:
        delta_X, delta_proj_X = checking(X, eigen_values, eigen_vectors)

    Q, R = linalg.qr(eigen_vectors)
    sigma2 = R @ np.diag(eigen_values) @ R.T
    sigma2_ = sigma2.diagonal()
    # X_recons = Q @ sigma2_ @ Q.T
    # delta_X = X_recons - X
    return sigma2_, Q


# def Schmidt_orthogonalization(eigenvector):
#     num = eigenvector.shape[1]
#     for i in range(num):
#         for i0 in range(i):
#             eigenvector[:, i] = eigenvector[:, i] - eigenvector[:, i0]*np.dot(eigenvector[:, i].transpose().conj(), eigenvector[:, i0])/(np.dot(eigenvector[:, i0].transpose().conj(),eigenvector[:, i0]))
#         eigenvector[:, i] = eigenvector[:, i]/np.linalg.norm(eigenvector[:, i])
#         print('i = {}'.format(i))
#     return eigenvector

