import librosa
import math
import numpy as np
import matplotlib.pyplot as plt

# 3.1 Projection
filename = "./Misirlou.wav"
audio, sr = librosa.load(filename, sr=16000)

spectrogram = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048)
# M represents the music file
# rows of M correspond to the frequencies
# columns of M correspond to the time
M = abs(spectrogram)
phase = spectrogram / (M + 2.2204e-16)

M.shape, spectrogram.shape

# read all the notes
# transform them to the spectrogram
# to represent each note
# choose relevant frequencies
# normalize them

# notes: 0 to 6.wav
note0to6 = []
for i in range(0, 7):
    filepath = "./notes_scale/"
    filename = "%s.wav" % i
    audio, sr = librosa.load(filepath + filename, sr=16000)
    spectrogram_note = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048)
    n = abs(spectrogram_note)
    middle = n[:, int(math.ceil(n.shape[1] / 2))]
    middle[middle < (max(middle) / 100)] = 0
    middle = middle / np.linalg.norm(middle)
    note0to6.append(middle)
# every column is one note
N = np.mat(note0to6).T
N.shape

# to compute the matrix W to let NW be nearest to M
# N W = M
# W = left_inv(N) M
# W = (N.T N)^-1 N.T M
W = np.linalg.inv(np.conj(N).T*N) * np.conj(N).T * M
np.savetxt('./result/problem1.csv', W, delimiter=',')

W[W < 0] = 0
M_hat = np.matmul(N, W)
M_hat.shape, phase.shape

# report the F2 error between M_hat and M
error_M = M - M_hat
F2 = np.square(np.linalg.norm(error_M))
print(F2)

signal_hat = librosa.istft(np.multiply(M_hat, phase), hop_length=256, center=False, win_length=2048)
librosa.output.write_wav("./result/resynthensized_proj.wav", signal_hat, sr=16000)


# 3.2 Optimization and non-negative decomposition

# report the formula for dE / dW
# E = 1/DT || M - NW ||^2
# dE/dW = 2/DT * (-1) * N.T (M-NW)
M.shape, N.shape, W.shape


def algorithm(yita, M, N):
    D, T = M.shape[0], M.shape[1]
    W = np.zeros([7, 1978])
    E_list = []
    for i in range(1000):
        E = 1 / (D * T) * np.sum(np.square(M - N * W))
        E_list.append(E)
        dE_dW = 2 / (D * T) * (-1) * np.conj(N).T * (M - N * W)
        W = W - yita * dE_dW
        W = np.maximum(W, 0)
    final_E = 1 / (D * T) * np.sum(np.square(M - N * W))
    return W, final_E, E_list


yita_list = [100, 1000, 10000, 100000]
W_list = []
E_final_list = []
E_list_list = []
for yita in yita_list:
    W, E, E_list = algorithm(yita, M, N)
    W_list.append(W)
    E_final_list.append(E)
    E_list_list.append(E_list)
    print(yita, E)

# plot E as iteration for all yitas
x = range(1000)
for i, y in enumerate(E_list_list):
    plt.plot(x, y)

plt.title("E and iterations for different learning rate")
plt.xlabel("iterations")
plt.ylabel("E")
plt.legend(["yita=%s" % i for i in yita_list])
plt.savefig("./result/problem3.2E_plot.png")
plt.show()

# In the figure, when yita = 100000, the E is minimized for the four yita.
# so the fourth W is the best choice here.
W_best = W_list[3]
np.savetxt('./result/problem2W.csv', W, delimiter = ',')

M_hat = N*W_best
signal_hat = librosa.istft(np.multiply(M_hat, phase), hop_length=256, center=False, win_length=2048)
librosa.output.write_wav("./result/resynthensized_nnproj.wav", signal_hat, sr=16000)