import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from problem3_adaboost import adaboost_train, adaboost_predict
from fobi_ica import FOBIICA
# from sklearn.decomposition import FastICA
from scipy import linalg
from utils import accuracy_score


def read_train_test_data(root_dir):
    def read_dir(subdir, subsubdir, label=None, root_dir=''):
        full_subdir = os.path.join(root_dir, subdir)

        img_dir = os.path.join(full_subdir, subsubdir)
        img_files = sorted(os.listdir(img_dir))
        img_files = [os.path.join(img_dir, f) for f in img_files]
        X = [np.asarray(Image.open(file).getdata()) for file in img_files]
        Y = len(X) * [label]
        return X, Y

    X_tr_pos, Y_tr_pos = read_dir('train', 'face', label=1., root_dir=root_dir)
    X_tr_neg, Y_tr_neg = read_dir('train', 'non-face', label=-1., root_dir=root_dir)
    X_te_pos, Y_te_pos = read_dir('test', 'face', label=1., root_dir=root_dir)
    X_te_neg, Y_te_neg = read_dir('test', 'non-face', label=-1., root_dir=root_dir)

    X_tr = np.array(X_tr_pos + X_tr_neg)
    Y_tr = np.array(Y_tr_pos + Y_tr_neg)
    X_te = np.array(X_te_pos + X_te_neg)
    Y_te = np.array(Y_te_pos + Y_te_neg)
    return X_tr, Y_tr, X_te, Y_te


def read_lfw1000(lfw_dir):
    filenames = sorted(os.listdir(lfw_dir))
    filenames = [os.path.join(lfw_dir, name) for name in filenames]
    X = np.zeros((19 * 19, len(filenames)), dtype=np.float64)
    for i, filename in enumerate(filenames):
        image = Image.open(filename).resize((19, 19))
        X[:, i] = np.asarray(image.getdata())
    return X


def adaboost_pca(X, X_tr, Y_tr, X_te, Y_te):
    U, sigma, VT = np.linalg.svd(X)
    # plt.figure('eigen_face')
    # eigenface1 = U[:, :1].reshape(19, 19)
    # plt.imshow(eigenface1, cmap='gray')
    # plt.tight_layout()

    k_list = [10, 30, 50]
    T_list = [10, 50, 150, 200]
    plt.figure('pca')
    for k in k_list:
        acc_train_list = []
        acc_test_list = []
        for T in T_list:
            print('k = {}, T = {}'.format(k, T))
            X_tr_weight = X_tr @ U[:, :k]
            X_te_weight = X_te @ U[:, :k]

            model = adaboost_train(X_tr_weight, Y_tr, T=T)

            pred_tr = adaboost_predict(model, X_tr_weight)
            acc_train = accuracy_score(Y_tr, pred_tr)
            print("acc_train: ", acc_train)
            acc_train_list.append(acc_train)

            pred_te = adaboost_predict(model, X_te_weight)
            acc_test = accuracy_score(Y_te, pred_te)
            print("acc_test: ", acc_test)
            acc_test_list.append(acc_test)

        plt.cla()
        plt.plot(T_list, 1 - np.array(acc_train_list), label='train error', c='b', marker='o')
        plt.plot(T_list, 1 - np.array(acc_test_list), label='test error', c='r', marker='o')
        plt.legend()
        plt.title('k={}'.format(k))
        plt.tight_layout()
        plt.savefig('results/pca_k{}.jpg'.format(k))
    return


def adaboost_ica(X, X_tr, Y_tr, X_te, Y_te):
    k_list = [10, 30, 50]
    T_list = [10, 50, 150, 200]
    plt.figure('ica')
    for k in k_list:
        acc_train_list = []
        acc_test_list = []
        for T in T_list:
            print('k = {}, T = {}'.format(k, T))
            # model = FastICA(n_components=k, random_state=0)
            model = FOBIICA(n_components=10)

            X_src = model.fit_transform(X)

            X_tr_weight = X_tr @ linalg.pinv(X_src).T
            X_te_weight = X_te @ linalg.pinv(X_src).T

            model = adaboost_train(X_tr_weight, Y_tr, T=T)

            pred_tr = adaboost_predict(model, X_tr_weight)
            acc_train = accuracy_score(Y_tr, pred_tr)
            print("acc_train: ", acc_train)
            acc_train_list.append(acc_train)

            pred_te = adaboost_predict(model, X_te_weight)
            acc_test = accuracy_score(Y_te, pred_te)
            print("acc_test: ", acc_test)
            acc_test_list.append(acc_test)

        plt.cla()
        plt.plot(T_list, 1 - np.array(acc_train_list), label='train error', c='b', marker='o')
        plt.plot(T_list, 1 - np.array(acc_test_list), label='test error', c='r', marker='o')
        plt.legend()
        plt.title('k={}'.format(k))
        plt.tight_layout()
        plt.savefig('results/ica_k{}.jpg'.format(k))
    return


def main():
    root_dir = 'data/hw2_materials_f21/problem3'
    X_tr, Y_tr, X_te, Y_te = read_train_test_data(root_dir)

    lfw_dir = 'data/hw2_materials_f21/problem3/lfw1000'
    X = read_lfw1000(lfw_dir)

    adaboost_pca(X, X_tr, Y_tr, X_te, Y_te)
    adaboost_ica(X, X_tr, Y_tr, X_te, Y_te)
    return


if __name__ == '__main__':
    main()
    print('done!')
