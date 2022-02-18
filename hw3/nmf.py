import numpy as np


def NMF_train(M, B_init, W_init, n_iter, update_B=True, update_W=True):
    B, W = B_init, W_init
    ones = np.ones_like(M)
    for i in range(n_iter):
        if update_B:
            B = B * ((M / (B @ W)) @ W.T) / (ones @ W.T)
        if update_W:
            W = W * (B.T @ (M / (B @ W))) / (B.T @ ones)
    return B, W


def separate_signals(M_mixed, W_init=None, B_speech=None, B_music=None, n_iter=500):
    _, n1 = B_speech.shape
    # _, n2 = B_music.shape
    B = np.concatenate([B_speech, B_music], axis=1)
    _, W = NMF_train(M_mixed, B, W_init, n_iter, update_B=False, update_W=True)
    W_speech = W[:n1, :]
    W_music = W[n1:, :]
    M_speech_rec = B_speech @ W_speech
    M_music_rec = B_music @ W_music
    return M_speech_rec, M_music_rec


def KL_divergence(A, B):
    out = A * np.log(A / B) - A + B
    out = out.sum()
    return out
