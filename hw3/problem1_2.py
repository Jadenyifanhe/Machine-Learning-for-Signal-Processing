import os
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x, (x - 1) ** 2 + 1


def main():
    # question1.2.1
    x_pos = np.array([0, 1, 2])
    x_neg = np.array([-2, -1, 3])
    y_pos = np.array(list(map(func, x_pos)))
    y_neg = np.array(list(map(func, x_neg)))
    label_pos = np.ones_like(x_pos)
    label_neg = -1 * np.ones_like(x_pos)

    xs = np.concatenate([x_pos, x_neg])
    ys = np.concatenate([y_pos, y_neg])
    labels = np.concatenate([label_pos, label_neg])

    plt.figure(1)
    plt.scatter(x_pos, np.zeros_like(x_pos), c='r', marker='o')
    plt.scatter(x_neg, np.zeros_like(x_neg), c='g', marker='o')
    plt.scatter(y_pos[:, 0], y_pos[:, 1], c='r', marker='*')
    plt.scatter(y_neg[:, 0], y_neg[:, 1], c='g', marker='*')

    # question1.2.2
    support_pos_idx = np.array([0, 2])
    support_neg_idx = np.array([4, 5])
    plt.scatter(ys[support_pos_idx, 0], ys[support_pos_idx, 1], c='none', marker='o', edgecolors='r', s=200)
    plt.scatter(ys[support_neg_idx, 0], ys[support_neg_idx, 1], c='none', marker='o', edgecolors='g', s=200)

    xx = np.arange(-2.1, 3.2, 0.1)
    yy_curve = np.array(list(map(func, xx)))
    yy_plane = 3.5 * np.ones_like(xx)
    plt.plot(yy_curve[:, 0], yy_curve[:, 1], linestyle='--', c='b')
    plt.plot(xx, yy_plane, c='y')
    plt.tight_layout()
    save_path = 'results/problem1/'
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'fig1.2.jpg')
    plt.savefig(filename)
    return


if __name__ == '__main__':
    main()
    print('done!')
