
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    data_root = 'data/hw2_materials_f21/problem3/lfw1000'
    filenames = sorted(os.listdir(data_root))
    filenames = [os.path.join(data_root, name) for name in filenames]

    image = Image.open(filenames[0])
    nrows, ncolumns = image.height, image.width

    # plt.imshow(image_mat, cmap='gray')

    X = np.zeros((nrows*ncolumns, len(filenames)), dtype=np.float64)
    for i, filename in enumerate(filenames):
        image = Image.open(filename)
        X[:, i] = np.asarray(image.getdata())

    U, sigma, VT = np.linalg.svd(X)
    save_dir = 'results/problem3/'
    os.makedirs(save_dir, exist_ok=True)
    csv_save_name = os.path.join(save_dir, 'eigenface.csv')
    np.savetxt(csv_save_name, U[:, :1], delimiter=',')

    plt.figure('eigen_face')
    eigenface1 = U[:, :1].reshape(nrows, ncolumns)
    plt.imshow(eigenface1, cmap='gray')
    fig_save_name = os.path.join(save_dir, 'eigenface1.jpg')
    plt.savefig(fig_save_name)

    theta = np.arccos(np.dot(U[0, :], U[1, :]) / (np.linalg.norm(U[0, :]) * np.linalg.norm(U[1, :]))) * 180 / np.pi
    print('theta = {} degree'.format(theta))

    k_error_list = []
    k_max = 100
    # plt.figure('eigen_faces')
    for k in range(k_max):
        eigenfacevector = U[:, k]
        eigenfaceimage = eigenfacevector.reshape(nrows, ncolumns)
        # plt.cla()
        # plt.imshow(eigenfaceimage, cmap='gray')
        # plt.tight_layout()
        # plt.pause(0.001)
        face_recons = reconstruct_face(U, sigma, VT, k+1)
        error = np.linalg.norm(face_recons - X, axis=0)
        error = error.mean()
        k_error_list.append((k+1, error))

    k_error = np.array(k_error_list)
    print('k = {}, error = {:.3f}'.format(k_error[-1, 0], k_error[-1, 1]))
    plt.figure('reconstruction error')
    plt.plot(k_error[:, 0], k_error[:, 1], marker='*')
    plt.xlabel("number of eigenfaces")
    plt.ylabel("mean reconstruction error")
    k_error_save_name = os.path.join(save_dir, 'k_error_pca.jpg')
    plt.savefig(k_error_save_name)
    return


def reconstruct_face(U, sigma, VT, k=1):
    faces = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]
    return faces


if __name__ == '__main__':
    main()
    print('done!')

