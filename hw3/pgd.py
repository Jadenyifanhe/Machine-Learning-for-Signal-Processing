import numpy as np


def proximal_gradient_descent(P, R, n_iterations=500, lamb=300):
    # Proximal Gradient Descent
    A = 2 * R.T @ R
    eig_vals = np.linalg.eigvals(A)
    L = eig_vals.max()  # L-Lipschitz condition

    M, N = R.shape
    x = np.zeros((N, 1))
    print('lamb = {}'.format(lamb))
    for i in range(n_iterations):
        grad = -2 * R.T @ (P - R @ x)
        z = x - grad / L
        x_new = np.zeros_like(x)
        for j in range(len(x)):
            if z[j] > lamb / L:
                x_new[j] = z[j] - lamb / L
            elif z[j] < -lamb / L:
                x_new[j] = z[j] + lamb / L
            else:
                x_new[j] = 0
        delta_x = np.linalg.norm(x_new - x)
        if i % 10 == 0 or i == n_iterations - 1:
            print('i = {}, delta_x = {:.3f}'.format(i, delta_x))
        x = x_new
    return x
