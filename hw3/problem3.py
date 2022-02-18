import os
import numpy as np
from PIL import Image
from sklearn import linear_model
from iht import iht
from pgd import proximal_gradient_descent
from utils import (read_csv, F2image, show_and_save, cal_rec_error)

SRC_DIR = 'data/hw3_materials/problem3/data/'
SAVE_DIR = 'results/problem3/'
ref_img_file = os.path.join(SRC_DIR, '../cassini128.png')
os.makedirs(SAVE_DIR, exist_ok=True)


def solve_lasso(P, R, solver='default'):
    if solver == 'default':
        lamb = 100
        n_samples = P.shape[0]
        clf = linear_model.Lasso(alpha=lamb / (2 * n_samples))
        clf.fit(R, P)
        F = clf.coef_
    elif solver == 'PGD':
        F = proximal_gradient_descent(P, R, n_iterations=500, lamb=50)
    else:
        raise ValueError('unsupported solver: {}'.format(solver))
    return F


def problem3_1():
    dims = [1024, 2048, 4096]
    p_filepath = os.path.join(SRC_DIR, 'P_{}.csv')
    r_filepath = os.path.join(SRC_DIR, 'R_{}.csv')
    for i, dim in enumerate(dims, 1):
        print('dim = {}'.format(dim))
        P = read_csv(p_filepath.format(dim))
        R = read_csv(r_filepath.format(dim))
        F = solve_lasso(P, R, solver='default')
        rec_image = F2image(F)
        # rec_image = (rec_image - rec_image.min())/rec_image.max()*255     # it seems extra
        filename = os.path.join(SAVE_DIR, 'rec_l1_{}.jpg'.format(dim))
        show_and_save(rec_image, num=i, filename=filename)
        src_img = np.array(Image.open(ref_img_file))
        error_total, error_mean = cal_rec_error(rec_image, src_img)
        print('error_total = {:.3f}, error_mean = {:.3f}'.format(error_total, error_mean))
    return


def problem3_2():
    def func3_2(P, R, max_iter=None):
        F = iht(P, R, max_iter=max_iter)
        image = F2image(F)
        return image

    dims = [1024, 2048, 4096]
    p_filepath = os.path.join(SRC_DIR, 'P_{}.csv')
    r_filepath = os.path.join(SRC_DIR, 'R_{}.csv')
    for i, dim in enumerate(dims, 1):
        print('dim = {}'.format(dim))
        P = read_csv(p_filepath.format(dim))
        R = read_csv(r_filepath.format(dim))
        rec_image = func3_2(P, R, max_iter=500)
        filename = os.path.join(SAVE_DIR, 'rec_iht_{}.jpg'.format(dim))
        show_and_save(rec_image, num=i, filename=filename)
        src_img = np.array(Image.open(ref_img_file))
        error_total, error_mean = cal_rec_error(rec_image, src_img)
        print('error_total = {:.3f}, error_mean = {:.3f}'.format(error_total, error_mean))
        print('')
    return


def main():
    problem3_1()
    #problem3_2()
    return


if __name__ == '__main__':
    main()
    print('done!')
