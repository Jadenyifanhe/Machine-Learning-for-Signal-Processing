import os
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = 'data/hw4materials/problem2'
SAVE_DIR = 'results/problem2'
os.makedirs(SAVE_DIR, exist_ok=True)


def read_data(filename=None):
    img = Image.open(filename)
    img = np.array(img)  # (288, 531)
    return img


def save_fig(img=None, ite=0):
    filename = os.path.join(SAVE_DIR, 'x1y1_ite_{}.jpg'.format(ite))
    plt.figure(1)
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.savefig(filename)
    return


def save_plot(y, ite=0):
    filename = os.path.join(SAVE_DIR, 'x2_ite_{}.jpg'.format(ite))
    plt.figure(2)
    plt.clf()
    plt.plot(y, marker='*')
    plt.tight_layout()
    plt.savefig(filename)
    return


def initialization(p_xy, w, h, w1, w2):
    # initialize p_x1y1, p_x2 and p_x2_cond_xy
    p_x1y1 = p_xy[:w1, :]
    # p_x1y1 = np.ones(shape=(w1, h))
    p_x1y1 = p_x1y1 / p_x1y1.sum()

    p_x2 = np.ones(w2)
    p_x2 = p_x2 / p_x2.sum()

    # joint distribution
    p_x2_cond_xy = np.ones(shape=(w2, w, h))
    for i in range(w):
        for j in range(h):
            p_x2_cond_xy[:, i, j] /= p_x2_cond_xy[:, i, j].sum()
    return p_x1y1, p_x2, p_x2_cond_xy


def EM(p_xy=None, w2=20, n_iter=100):
    w, h = p_xy.shape
    w1, h1 = w - w2, h

    p_x1y1, p_x2, p_x2_cond_xy = initialization(p_xy, w, h, w1, w2)

    for ite in range(n_iter):
        # E step
        p_x2_cond_xy_pre = copy.deepcopy(p_x2_cond_xy)
        for i2 in range(w2):
            # print('i2 = {}'.format(i2))
            for i in range(w):
                for j in range(h):
                    if i < i2 or i > i2 + w1 - 1:
                        p_x2_cond_xy[i2, i, j] = 0
                    else:
                        tmp = 0
                        low = max(i - w1 + 1, 0)
                        high = min(i, w2 - 1)
                        for ii2 in range(low, high + 1):
                            tmp += p_x2[ii2] * p_x1y1[i - ii2, j]
                        p_x2_cond_xy[i2, i, j] = p_x2[i2] * p_x1y1[i - i2, j] / (tmp + np.finfo(np.float64).eps)

        # M step
        p_x1y1_pre = copy.deepcopy(p_x1y1)
        for i1 in range(w1):
            for j1 in range(h1):
                tmp = 0
                low = i1
                high = min(i1 + w2 - 1, w)
                for ii in range(low, high + 1):
                    tmp += p_xy[ii, j1] * p_x2_cond_xy[ii - i1, ii, j1]
                p_x1y1[i1, j1] = tmp

        p_x2_pre = copy.deepcopy(p_x2)
        for i2 in range(w2):
            tmp = 0
            for i in range(w):
                for j in range(h):
                    tmp += p_xy[i, j] * p_x2_cond_xy[i2, i, j]
            p_x2[i2] = tmp

        save_fig(img=np.transpose(p_x1y1, (1, 0)), ite=ite)
        save_plot(y=p_x2, ite=ite)

        # p_x1y1 = p_x1y1 / p_x1y1.sum()
        # p_x2 = p_x2 / p_x2.sum()
        # for i in range(w):
        #     for j in range(h):
        #         if p_x2_cond_xy[:, i, j].sum() <= 0:
        #             p_x2_cond_xy[:, i, j] = 0
        #         else:
        #             p_x2_cond_xy[:, i, j] /= p_x2_cond_xy[:, i, j].sum()

        print('iteration: {}, delta(p_x2_cond_xy) = {:.3e}, delta(p_x1y1) = {:.3e}, delta(p_x2) = {:.3e}'.format(
            ite,
            np.linalg.norm(p_x2_cond_xy - p_x2_cond_xy_pre),
            np.linalg.norm(p_x1y1 - p_x1y1_pre),
            np.linalg.norm(p_x2 - p_x2_pre)
        ))
    return p_x1y1, p_x2


def main():
    filename = os.path.join(SRC_DIR, 'carblurred.png')
    img = read_data(filename)
    p_xy = np.transpose(img, (1, 0)).astype(np.float64)
    # p_xy = p_xy / p_xy.sum()
    p_x1y1, p_x2 = EM(p_xy=p_xy, n_iter=5000)
    return


if __name__ == '__main__':
    main()
    print('done!')
