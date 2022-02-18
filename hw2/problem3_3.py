
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fobi_ica import FOBIICA
# from sklearn.decomposition import FastICA


def main():
    data_root = 'data/hw2_materials_f21/problem3/lfw1000'
    filenames = sorted(os.listdir(data_root))
    filenames = [os.path.join(data_root, name) for name in filenames]

    image = Image.open(filenames[0])
    nrows, ncolumns = image.height, image.width

    X = np.zeros((len(filenames), nrows*ncolumns), dtype=np.float64)
    for i, filename in enumerate(filenames):
        image = Image.open(filename)
        X[i, :] = np.asarray(image.getdata())

    model = FOBIICA(n_components=100)
    # model = FastICA(n_components=100, random_state=0)
    X_transformed = model.fit_transform(X.T).T
    ica_face1 = X_transformed[0, :].reshape(nrows, ncolumns)
    theta = np.arccos(np.dot(model.mixing_[:, 0], model.mixing_[:, 1]) / (np.linalg.norm(model.mixing_[:, 0]) * np.linalg.norm(model.mixing_[:, 1]))) * 180 / np.pi
    print('theta = {} degree'.format(theta))

    save_dir = 'results/problem3/'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure('ica_face1')
    plt.imshow(ica_face1, cmap='gray')
    # plt.imshow(ica_face1)
    fig_save_name = os.path.join(save_dir, 'ica_face1.jpg')
    plt.savefig(fig_save_name)

    # plt.figure('ica_faces')
    # for i in range(len(X_transformed)):
    #     ica_face = X_transformed[i].reshape(nrows, ncolumns)
    #     plt.cla()
    #     plt.imshow(ica_face, cmap='gray')
    #     plt.tight_layout()
    #     plt.pause(0.001)

    k_max = 100
    k_error_list = []
    for k in range(1, k_max+1):
        print('k = {}'.format(k))
        model = FOBIICA(n_components=k, random_state=0)
        # model = FastICA(n_components=k, random_state=0)
        X_transformed = model.fit_transform(X.T).T
        X_recons = model.inverse_transform(X_transformed.T).T
        X_error = X_recons - X
        X_error = np.linalg.norm(X_error, axis=1)
        X_error = X_error.mean()
        k_error_list.append((k, X_error))

    k_error = np.array(k_error_list)
    print('k = {}, error = {:.3f}'.format(k_error[-1, 0], k_error[-1, 1]))
    plt.figure('reconstruction error')
    plt.xlabel("number of ICA faces")
    plt.ylabel("mean reconstruction error")
    plt.plot(k_error[:, 0], k_error[:, 1], marker='o', c='r')
    plt.tight_layout()
    k_error_save_name = os.path.join(save_dir, 'k_error_ica.jpg')
    plt.savefig(k_error_save_name)
    return


if __name__ == '__main__':
    main()
    print('done!')
