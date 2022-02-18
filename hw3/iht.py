import numpy as np
import scipy.linalg as linalg


def cal_measurement_error(P, R, F):
    error = P - R @ F
    error = linalg.norm(error)
    return error


def project2feasible(F, K=None):
    Fk = np.zeros_like(F)
    idx = np.argsort(np.fabs(F.squeeze()))
    idx = idx[::-1]
    Fk[idx[:K]] = F[idx[:K]]
    return Fk


def solution1(P, R, F0, alpha, K, max_iter=None):
    R_hat = R * alpha
    P_hat = P * alpha
    F = F0
    error = cal_measurement_error(P, R, F)
    print('i = {}, error = {:.3f}'.format(0, error))
    for i in range(1, max_iter + 1):
        F_unconstrained = F + R_hat.T @ (P_hat - R_hat @ F)
        F = project2feasible(F_unconstrained, K=K)
        if i % 10 == 0:
            error = cal_measurement_error(P, R, F)
            print('i = {}, error = {:.3f}'.format(i, error))
    return F


def solution2(P, R, F0, alpha, K=None, max_iter=None):
    R_hat = R * alpha
    F_hat = F0 / alpha

    error = cal_measurement_error(P, R, F0)
    print('i = {}, error = {:.3f}'.format(0, error))
    for i in range(1, max_iter + 1):
        F_hat_unconstrained = F_hat + R_hat.T @ (P - R_hat @ F_hat)
        F_hat = project2feasible(F_hat_unconstrained, K=K)
        F = alpha * F_hat

        if i % 10 == 0:
            error = cal_measurement_error(P, R, F)
            print('i = {}, error = {:.3f}'.format(i, error))
    return F


def iht(P, R, max_iter=None):
    M, N = R.shape
    F0 = np.zeros((N, 1))
    K = np.round(np.max([M / 4, N / 10])).astype(int)
    norm_R = linalg.norm(R)
    alpha = 1.0 / norm_R
    F = solution1(P, R, F0, alpha, K, max_iter=max_iter)
    # F = solution2(P, R, F0, alpha, K, max_iter=max_iter)
    return F
